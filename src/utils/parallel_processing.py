"""
Parallel Processing Module for VeritasFinancial Banking Fraud Detection System

This module provides comprehensive parallel processing utilities for:
- Multi-core CPU parallelization
- GPU parallelization (CUDA, MPS)
- Distributed computing (Ray, Dask)
- Batch processing for large datasets
- Progress tracking and monitoring
- Resource management
- Error handling and recovery

Author: VeritasFinancial Data Science Team
Version: 1.0.0
"""

import multiprocessing as mp
from multiprocessing import Pool, Process, Queue, Manager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import ray
import dask
from dask.distributed import Client, LocalCluster
import torch
import numpy as np
import pandas as pd
from typing import Any, Callable, List, Dict, Optional, Union, Iterator, Tuple
from functools import partial
import logging
import time
from tqdm import tqdm
import psutil
import gc
from dataclasses import dataclass
from enum import Enum
import signal
import traceback

# Configure logging
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Enum for different processing modes."""
    CPU = "cpu"
    GPU = "gpu"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


class ResourceManager:
    """
    Resource Manager for parallel processing.
    
    Manages CPU cores, GPU memory, and system resources to prevent
    over-subscription and ensure efficient parallelization.
    """
    
    def __init__(self, 
                 max_cpu_cores: Optional[int] = None,
                 max_gpu_memory: Optional[float] = None,
                 max_ram_usage: float = 0.8):
        """
        Initialize resource manager.
        
        Args:
            max_cpu_cores: Maximum CPU cores to use (None = all available)
            max_gpu_memory: Maximum GPU memory to use in GB
            max_ram_usage: Maximum RAM usage ratio (0.0-1.0)
        """
        self.max_cpu_cores = max_cpu_cores or mp.cpu_count()
        self.max_gpu_memory = max_gpu_memory
        self.max_ram_usage = max_ram_usage
        
        # Available resources
        self.total_ram = psutil.virtual_memory().total / (1024**3)  # GB
        self.available_ram = self.total_ram * max_ram_usage
        
        # GPU resources
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_memory = [
                torch.cuda.get_device_properties(i).total_memory / (1024**3)
                for i in range(self.gpu_count)
            ]
        else:
            self.gpu_count = 0
            self.gpu_memory = []
        
        # Resource locks
        self.cpu_lock = threading.Lock()
        self.gpu_locks = [threading.Lock() for _ in range(self.gpu_count)]
        
        logger.info(f"Resource Manager initialized: "
                   f"CPU cores={self.max_cpu_cores}, "
                   f"RAM={self.available_ram:.1f}GB, "
                   f"GPUs={self.gpu_count}")
    
    def get_cpu_cores(self, requested: Optional[int] = None) -> int:
        """
        Get available CPU cores.
        
        Args:
            requested: Number of cores requested
            
        Returns:
            Number of cores to use
        """
        with self.cpu_lock:
            if requested:
                return min(requested, self.max_cpu_cores)
            return self.max_cpu_cores
    
    def get_gpu_device(self, memory_required: float = 0) -> Optional[int]:
        """
        Get available GPU device.
        
        Args:
            memory_required: Required GPU memory in GB
            
        Returns:
            GPU device index or None if no GPU available
        """
        if not self.gpu_available:
            return None
        
        for i in range(self.gpu_count):
            if self.gpu_locks[i].acquire(blocking=False):
                try:
                    # Check available memory
                    if memory_required > 0:
                        free_memory = self._get_free_gpu_memory(i)
                        if free_memory < memory_required:
                            self.gpu_locks[i].release()
                            continue
                    
                    return i
                except:
                    self.gpu_locks[i].release()
                    continue
        
        return None
    
    def _get_free_gpu_memory(self, device: int) -> float:
        """Get free memory on GPU device."""
        try:
            torch.cuda.set_device(device)
            return torch.cuda.memory_reserved(device) / (1024**3)
        except:
            return 0
    
    def release_gpu(self, device: int):
        """Release GPU device lock."""
        if 0 <= device < len(self.gpu_locks):
            self.gpu_locks[device].release()
    
    def check_memory_available(self, required_memory: float) -> bool:
        """Check if sufficient RAM is available."""
        current_usage = psutil.virtual_memory().percent / 100
        return (1 - current_usage) * self.total_ram >= required_memory


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    mode: ProcessingMode = ProcessingMode.CPU
    num_workers: Optional[int] = None
    batch_size: int = 1000
    prefetch_factor: int = 2
    timeout: Optional[float] = None
    use_progress_bar: bool = True
    error_handling: str = 'raise'  # 'raise', 'skip', 'log'
    max_retries: int = 3
    retry_delay: float = 1.0
    pin_memory: bool = False
    num_gpus: Optional[int] = None
    distributed_address: Optional[str] = None


class ParallelProcessor:
    """
    Main parallel processing class for VeritasFinancial.
    
    Supports:
    - Multi-core CPU processing
    - GPU acceleration
    - Distributed computing with Ray/Dask
    - Batch processing with progress tracking
    - Automatic resource management
    - Error recovery
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel processor.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config or ParallelConfig()
        self.resource_manager = ResourceManager()
        self._workers = []
        self._ray_client = None
        self._dask_client = None
        self._stop_event = threading.Event()
        
        # Adjust workers based on mode
        if self.config.num_workers is None:
            if self.config.mode == ProcessingMode.CPU:
                self.config.num_workers = self.resource_manager.max_cpu_cores - 1
            elif self.config.mode == ProcessingMode.GPU:
                self.config.num_workers = self.resource_manager.gpu_count
            else:
                self.config.num_workers = 4
        
        logger.info(f"ParallelProcessor initialized with mode={self.config.mode}, "
                   f"workers={self.config.num_workers}")
    
    def map(self, 
            func: Callable, 
            items: List[Any], 
            **kwargs) -> List[Any]:
        """
        Parallel map operation.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            **kwargs: Additional arguments for the function
            
        Returns:
            List of results in the same order as input
        """
        if self.config.mode == ProcessingMode.CPU:
            return self._cpu_map(func, items, **kwargs)
        elif self.config.mode == ProcessingMode.GPU:
            return self._gpu_map(func, items, **kwargs)
        elif self.config.mode == ProcessingMode.DISTRIBUTED:
            return self._distributed_map(func, items, **kwargs)
        else:
            return self._hybrid_map(func, items, **kwargs)
    
    def _cpu_map(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """
        CPU-based parallel processing.
        
        Uses multiprocessing.Pool for true parallelism across CPU cores.
        """
        # Prepare items with kwargs
        if kwargs:
            func = partial(func, **kwargs)
        
        results = []
        failed_items = []
        
        # Use ProcessPoolExecutor for better control
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(func, item) for item in items]
            
            # Collect results with progress bar
            iterator = tqdm(futures, desc="Processing", 
                           disable=not self.config.use_progress_bar)
            
            for i, future in enumerate(iterator):
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                except Exception as e:
                    if self.config.error_handling == 'raise':
                        raise
                    elif self.config.error_handling == 'log':
                        logger.error(f"Error processing item {i}: {str(e)}")
                        failed_items.append((i, str(e)))
                        results.append(None)
                    else:  # skip
                        failed_items.append((i, str(e)))
                        continue
        
        if failed_items:
            logger.warning(f"Completed with {len(failed_items)} failed items")
        
        return results
    
    def _gpu_map(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """
        GPU-based parallel processing.
        
        Distributes work across available GPUs.
        """
        if not torch.cuda.is_available():
            logger.warning("GPU not available, falling back to CPU")
            return self._cpu_map(func, items, **kwargs)
        
        # Split items into batches for each GPU
        num_gpus = self.config.num_gpus or self.resource_manager.gpu_count
        batch_size = len(items) // num_gpus + 1
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        results = [None] * len(items)
        
        def process_batch(gpu_id, batch_items, batch_indices):
            """Process a batch on specific GPU."""
            try:
                # Set GPU device
                torch.cuda.set_device(gpu_id)
                
                # Process items
                batch_results = []
                for item in batch_items:
                    result = func(item, **kwargs, device=gpu_id)
                    batch_results.append(result)
                
                # Return results with indices
                return batch_indices, batch_results
                
            except Exception as e:
                logger.error(f"GPU {gpu_id} error: {str(e)}")
                return batch_indices, [None] * len(batch_items)
            finally:
                self.resource_manager.release_gpu(gpu_id)
        
        # Launch GPU processes
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            start_idx = 0
            
            for i, batch in enumerate(batches):
                if not batch:
                    continue
                    
                gpu_id = self.resource_manager.get_gpu_device()
                if gpu_id is None:
                    # Fallback to CPU for this batch
                    batch_results = [func(item, **kwargs) for item in batch]
                    for j, result in enumerate(batch_results):
                        results[start_idx + j] = result
                else:
                    indices = list(range(start_idx, start_idx + len(batch)))
                    future = executor.submit(process_batch, gpu_id, batch, indices)
                    futures.append(future)
                
                start_idx += len(batch)
            
            # Collect results
            for future in futures:
                indices, batch_results = future.result()
                for idx, result in zip(indices, batch_results):
                    results[idx] = result
        
        return results
    
    def batch_process(self,
                      func: Callable,
                      data_iterator: Iterator,
                      batch_size: Optional[int] = None) -> Iterator:
        """
        Process data in batches with parallel processing.
        
        Useful for large datasets that don't fit in memory.
        
        Args:
            func: Function to apply to each batch
            data_iterator: Iterator yielding data items
            batch_size: Size of batches (default from config)
            
        Yields:
            Processed results
        """
        batch_size = batch_size or self.config.batch_size
        current_batch = []
        
        for item in data_iterator:
            current_batch.append(item)
            
            if len(current_batch) >= batch_size:
                # Process batch in parallel
                results = self.map(func, current_batch)
                
                # Yield results one by one
                for result in results:
                    yield result
                
                current_batch = []
                
                # Clean up memory
                gc.collect()
        
        # Process remaining items
        if current_batch:
            results = self.map(func, current_batch)
            for result in results:
                yield result
    
    def parallel_dataframe(self,
                          df: pd.DataFrame,
                          func: Callable,
                          axis: int = 0,
                          **kwargs) -> pd.DataFrame:
        """
        Parallel processing on pandas DataFrame.
        
        Args:
            df: DataFrame to process
            func: Function to apply
            axis: 0 for rows, 1 for columns
            **kwargs: Additional arguments
            
        Returns:
            Processed DataFrame
        """
        if axis == 0:
            # Process rows
            chunks = np.array_split(df, self.config.num_workers)
            
            def process_chunk(chunk):
                return chunk.apply(func, axis=1, **kwargs)
            
            results = self.map(process_chunk, chunks)
            return pd.concat(results)
        
        else:
            # Process columns
            columns = list(df.columns)
            chunks = [columns[i::self.config.num_workers] 
                     for i in range(self.config.num_workers)]
            
            def process_columns(col_list):
                return df[col_list].apply(func, **kwargs)
            
            results = self.map(process_columns, chunks)
            return pd.concat(results, axis=1)
    
    def parallel_apply(self,
                       data: Union[pd.DataFrame, np.ndarray, List],
                       func: Callable,
                       **kwargs) -> Union[pd.DataFrame, np.ndarray, List]:
        """
        Generic parallel apply function.
        
        Automatically handles different data types.
        """
        if isinstance(data, pd.DataFrame):
            return self.parallel_dataframe(data, func, **kwargs)
        
        elif isinstance(data, np.ndarray):
            # Process numpy array
            chunks = np.array_split(data, self.config.num_workers)
            results = self.map(func, chunks, **kwargs)
            return np.concatenate(results)
        
        elif isinstance(data, list):
            # Process list
            chunk_size = len(data) // self.config.num_workers + 1
            chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            results = self.map(func, chunks, **kwargs)
            return [item for sublist in results for item in sublist]
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def pipeline(self,
                 stages: List[Callable],
                 initial_data: Any) -> Any:
        """
        Parallel pipeline processing.
        
        Processes data through multiple stages with parallel execution
        where possible.
        
        Args:
            stages: List of processing functions
            initial_data: Initial data to process
            
        Returns:
            Processed data after all stages
        """
        data = initial_data
        
        for i, stage in enumerate(stages):
            logger.info(f"Pipeline stage {i+1}/{len(stages)}")
            
            if isinstance(data, (list, np.ndarray, pd.DataFrame)):
                # Parallelize if data is collection
                data = self.map(stage, data)
            else:
                # Single item processing
                data = stage(data)
        
        return data
    
    def cleanup(self):
        """Clean up resources."""
        self._stop_event.set()
        
        # Stop distributed clients
        if self._ray_client:
            ray.shutdown()
        
        if self._dask_client:
            self._dask_client.close()
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5)
        
        logger.info("ParallelProcessor cleaned up")


class ChunkedDataLoader:
    """
    Memory-efficient data loader that loads and processes data in chunks.
    
    Useful for large datasets that don't fit in memory.
    """
    
    def __init__(self,
                 file_path: str,
                 chunk_size: int = 10000,
                 parser: Optional[Callable] = None):
        """
        Initialize chunked data loader.
        
        Args:
            file_path: Path to data file
            chunk_size: Number of rows per chunk
            parser: Function to parse each chunk
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.parser = parser or self._default_parser
    
    def __iter__(self) -> Iterator:
        """Iterate over chunks."""
        if self.file_path.endswith('.csv'):
            return self._iter_csv()
        elif self.file_path.endswith('.parquet'):
            return self._iter_parquet()
        else:
            return self._iter_generic()
    
    def _iter_csv(self) -> Iterator:
        """Iterate over CSV file in chunks."""
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            yield self.parser(chunk)
    
    def _iter_parquet(self) -> Iterator:
        """Iterate over Parquet file in chunks."""
        parquet_file = pq.ParquetFile(self.file_path)
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            df = batch.to_pandas()
            yield self.parser(df)
    
    def _iter_generic(self) -> Iterator:
        """Generic file iterator."""
        with open(self.file_path, 'r') as f:
            chunk = []
            for line in f:
                chunk.append(line)
                if len(chunk) >= self.chunk_size:
                    yield self.parser(chunk)
                    chunk = []
            
            if chunk:
                yield self.parser(chunk)
    
    def _default_parser(self, chunk: Any) -> Any:
        """Default parser - returns chunk as is."""
        return chunk


class ProgressTracker:
    """
    Progress tracking for parallel processing.
    
    Provides real-time progress updates with ETA and speed.
    """
    
    def __init__(self, total: int, desc: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            desc: Description for progress bar
        """
        self.total = total
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.pbar = tqdm(total=total, desc=desc)
    
    def update(self, n: int = 1):
        """Update progress."""
        with self.lock:
            self.completed += n
            self.pbar.update(n)
    
    @property
    def progress(self) -> float:
        """Get progress percentage."""
        return self.completed / self.total
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_remaining(self) -> float:
        """Get estimated remaining time in seconds."""
        if self.completed == 0:
            return float('inf')
        return (self.elapsed_time / self.completed) * (self.total - self.completed)
    
    def close(self):
        """Close progress tracker."""
        self.pbar.close()


# Convenience functions
def parallel_map(func: Callable, items: List[Any], **kwargs) -> List[Any]:
    """Convenience function for parallel map."""
    processor = ParallelProcessor()
    return processor.map(func, items, **kwargs)


def parallel_apply(data: Any, func: Callable, **kwargs) -> Any:
    """Convenience function for parallel apply."""
    processor = ParallelProcessor()
    return processor.parallel_apply(data, func, **kwargs)


def batch_process(func: Callable, data: Iterator, **kwargs) -> Iterator:
    """Convenience function for batch processing."""
    processor = ParallelProcessor()
    return processor.batch_process(func, data, **kwargs)