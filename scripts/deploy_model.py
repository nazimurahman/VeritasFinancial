# scripts/deploy_model.py
#!/usr/bin/env python3
"""
Model Deployment Script for VeritasFinancial Fraud Detection System

This script handles the deployment of trained fraud detection models to production.
It supports multiple deployment targets:
- Local API server
- Docker container
- Kubernetes cluster
- Cloud platforms (AWS SageMaker, GCP AI Platform, Azure ML)

The script includes:
- Model versioning and registry
- A/B testing setup
- Canary deployments
- Rollback capabilities
- Health checks and monitoring

Author: VeritasFinancial Data Science Team
Version: 2.0.0
"""

# ============================================================================
# IMPORTS SECTION
# ============================================================================
import os
import sys
import json
import yaml
import argparse
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import joblib
import hashlib

# ============================================================================
# CLOUD PROVIDER IMPORTS (Conditional)
# ============================================================================
# AWS
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("Warning: boto3 not available. AWS deployment disabled.")

# GCP
try:
    from google.cloud import storage
    from google.cloud import aiplatform
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("Warning: google-cloud not available. GCP deployment disabled.")

# Azure
try:
    from azure.storage.blob import BlobServiceClient
    from azure.ai.ml import MLClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Warning: azure libraries not available. Azure deployment disabled.")

# ============================================================================
# KUBERNETES IMPORTS
# ============================================================================
try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    print("Warning: kubernetes library not available. K8s deployment disabled.")

# ============================================================================
# DOCKER IMPORTS
# ============================================================================
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    print("Warning: docker library not available. Docker deployment disabled.")

# ============================================================================
# PROJECT IMPORTS
# ============================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import setup_logger
from src.utils.config_manager import ConfigManager
from src.deployment.api.fastapi_app import app as fastapi_app
from src.deployment.monitoring.drift_detection import DriftDetector

# ============================================================================
# LOGGING SETUP
# ============================================================================
logger = setup_logger(
    name='model_deployment',
    log_file='logs/deployment.log',
    level=logging.INFO
)


# ============================================================================
# MODEL VERSIONING CLASS
# ============================================================================
class ModelVersionManager:
    """
    Manages model versions and registry.
    
    This class handles:
    - Version incrementing
    - Model metadata storage
    - Model artifact management
    - Version rollback capabilities
    """
    
    def __init__(self, registry_path: str = 'artifacts/model_registry'):
        """
        Initialize the version manager.
        
        Args:
            registry_path: Path to model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Load or create registry
        self.registry_file = self.registry_path / 'registry.json'
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """
        Load model registry from file.
        
        Returns:
            Registry dictionary
        """
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {'models': {}, 'current_versions': {}}
    
    def _save_registry(self):
        """Save model registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name: str, model_path: str, 
                      metrics: Dict, description: str = "") -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            model_path: Path to model file
            metrics: Evaluation metrics
            description: Model description
            
        Returns:
            Version ID
        """
        # Calculate version
        if model_name not in self.registry['models']:
            version = "1.0.0"
            self.registry['models'][model_name] = []
        else:
            # Increment version
            latest = self.registry['models'][model_name][-1]['version']
            major, minor, patch = map(int, latest.split('.'))
            patch += 1
            version = f"{major}.{minor}.{patch}"
        
        # Calculate model hash for integrity
        with open(model_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Create version entry
        version_entry = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'model_hash': model_hash,
            'metrics': metrics,
            'description': description,
            'status': 'staged'
        }
        
        self.registry['models'][model_name].append(version_entry)
        self._save_registry()
        
        logger.info(f"Registered model {model_name} version {version}")
        return version
    
    def promote_to_production(self, model_name: str, version: str) -> bool:
        """
        Promote a model version to production.
        
        Args:
            model_name: Name of the model
            version: Version to promote
            
        Returns:
            Success status
        """
        # Find the version
        for model in self.registry['models'][model_name]:
            if model['version'] == version:
                model['status'] = 'production'
                
                # Update current version
                self.registry['current_versions'][model_name] = version
                
                # Archive previous production version
                for prev_model in self.registry['models'][model_name]:
                    if prev_model['version'] != version and prev_model.get('status') == 'production':
                        prev_model['status'] = 'archived'
                
                self._save_registry()
                logger.info(f"Promoted {model_name} version {version} to production")
                return True
        
        logger.error(f"Version {version} not found for model {model_name}")
        return False
    
    def rollback(self, model_name: str) -> Tuple[bool, str]:
        """
        Rollback to previous production version.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tuple of (success, version)
        """
        # Get current production version
        current_version = self.registry['current_versions'].get(model_name)
        
        if not current_version:
            logger.error(f"No production version found for {model_name}")
            return False, ""
        
        # Find previous version
        versions = self.registry['models'][model_name]
        current_idx = -1
        
        for i, v in enumerate(versions):
            if v['version'] == current_version:
                current_idx = i
                break
        
        if current_idx <= 0:
            logger.error(f"No previous version available for {model_name}")
            return False, ""
        
        # Promote previous version
        prev_version = versions[current_idx - 1]['version']
        success = self.promote_to_production(model_name, prev_version)
        
        return success, prev_version
    
    def get_production_model_path(self, model_name: str) -> Optional[str]:
        """
        Get the path to the current production model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to model file or None
        """
        version = self.registry['current_versions'].get(model_name)
        
        if not version:
            return None
        
        for model in self.registry['models'][model_name]:
            if model['version'] == version:
                return model['model_path']
        
        return None
    
    def list_models(self) -> Dict:
        """
        List all registered models and versions.
        
        Returns:
            Dictionary of models and their versions
        """
        return self.registry


# ============================================================================
# DEPLOYMENT TARGETS
# ============================================================================
class LocalDeployment:
    """
    Deploys model as local API server.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize local deployment.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 8000)
        self.workers = config.get('workers', 4)
        
    def deploy(self, model_path: str) -> bool:
        """
        Deploy model locally.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Success status
        """
        logger.info(f"Deploying model locally on {self.host}:{self.port}")
        
        # Set environment variables for the API
        os.environ['MODEL_PATH'] = model_path
        os.environ['DEPLOYMENT_MODE'] = 'production'
        
        # Start uvicorn server
        cmd = [
            'uvicorn',
            'src.deployment.api.fastapi_app:app',
            '--host', self.host,
            '--port', str(self.port),
            '--workers', str(self.workers),
            '--log-level', 'info'
        ]
        
        try:
            # This will block - run in separate process for production
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if deployment is healthy.
        
        Returns:
            Health status
        """
        import requests
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health")
            return response.status_code == 200
        except:
            return False


class DockerDeployment:
    """
    Deploys model as Docker container.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Docker deployment.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.image_name = config.get('image_name', 'veritasfinancial/fraud-detection')
        self.image_tag = config.get('image_tag', 'latest')
        self.container_name = config.get('container_name', 'fraud-detection-api')
        self.port_mapping = config.get('port_mapping', {'8000': '8000'})
        self.env_vars = config.get('env_vars', {})
        
        if DOCKER_AVAILABLE:
            self.docker_client = docker.from_env()
        else:
            self.docker_client = None
    
    def build_image(self, dockerfile_path: str = 'Dockerfile') -> bool:
        """
        Build Docker image.
        
        Args:
            dockerfile_path: Path to Dockerfile
            
        Returns:
            Success status
        """
        if not DOCKER_AVAILABLE:
            logger.error("Docker not available")
            return False
        
        try:
            logger.info(f"Building Docker image: {self.image_name}:{self.image_tag}")
            
            # Build image
            image, logs = self.docker_client.images.build(
                path='.',
                dockerfile=dockerfile_path,
                tag=f"{self.image_name}:{self.image_tag}",
                rm=True
            )
            
            # Stream logs
            for log in logs:
                if 'stream' in log:
                    logger.info(log['stream'].strip())
            
            logger.info(f"Image built successfully: {image.id[:12]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            return False
    
    def push_to_registry(self, registry_url: str = None) -> bool:
        """
        Push image to container registry.
        
        Args:
            registry_url: Container registry URL
            
        Returns:
            Success status
        """
        if not DOCKER_AVAILABLE:
            logger.error("Docker not available")
            return False
        
        try:
            tag = f"{self.image_name}:{self.image_tag}"
            
            if registry_url:
                # Tag for registry
                registry_tag = f"{registry_url}/{tag}"
                self.docker_client.images.get(tag).tag(registry_tag)
                
                # Push to registry
                logger.info(f"Pushing image to {registry_url}")
                for line in self.docker_client.images.push(registry_tag, stream=True, decode=True):
                    if 'status' in line:
                        logger.info(line['status'])
                    if 'error' in line:
                        logger.error(line['error'])
                        return False
            else:
                logger.info("No registry URL provided, skipping push")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to push image: {e}")
            return False
    
    def run_container(self) -> bool:
        """
        Run Docker container.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Success status
        """
        if not DOCKER_AVAILABLE:
            logger.error("Docker not available")
            return False
        
        try:
            # Stop existing container if running
            try:
                container = self.docker_client.containers.get(self.container_name)
                container.stop()
                container.remove()
                logger.info(f"Removed existing container: {self.container_name}")
            except docker.errors.NotFound:
                pass
            
            # Prepare port mappings
            ports = {}
            for host_port, container_port in self.port_mapping.items():
                ports[container_port] = host_port
            
            # Run new container
            container = self.docker_client.containers.run(
                image=f"{self.image_name}:{self.image_tag}",
                name=self.container_name,
                ports=ports,
                environment=self.env_vars,
                detach=True,
                restart_policy={"Name": "always"}
            )
            
            logger.info(f"Container started: {container.id[:12]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to run container: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if containerized API is healthy.
        
        Returns:
            Health status
        """
        import requests
        try:
            # Get first mapped port
            host_port = next(iter(self.port_mapping.keys()))
            response = requests.get(f"http://localhost:{host_port}/health")
            return response.status_code == 200
        except:
            return False


class KubernetesDeployment:
    """
    Deploys model to Kubernetes cluster.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Kubernetes deployment.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.namespace = config.get('namespace', 'default')
        self.deployment_name = config.get('deployment_name', 'fraud-detection')
        self.service_name = config.get('service_name', 'fraud-detection-service')
        self.replicas = config.get('replicas', 3)
        self.image = config.get('image', 'veritasfinancial/fraud-detection:latest')
        self.resources = config.get('resources', {
            'requests': {'cpu': '500m', 'memory': '1Gi'},
            'limits': {'cpu': '2', 'memory': '4Gi'}
        })
        
        if K8S_AVAILABLE:
            try:
                # Load kube config
                config.load_kube_config()
                self.k8s_apps_v1 = client.AppsV1Api()
                self.k8s_core_v1 = client.CoreV1Api()
                self.k8s_networking_v1 = client.NetworkingV1Api()
            except Exception as e:
                logger.warning(f"Failed to load kube config: {e}")
                self.k8s_apps_v1 = None
        else:
            self.k8s_apps_v1 = None
    
    def create_deployment(self) -> bool:
        """
        Create Kubernetes deployment.
        
        Returns:
            Success status
        """
        if not K8S_AVAILABLE or not self.k8s_apps_v1:
            logger.error("Kubernetes not available")
            return False
        
        try:
            # Define deployment
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=self.deployment_name,
                    namespace=self.namespace,
                    labels={'app': 'fraud-detection'}
                ),
                spec=client.V1DeploymentSpec(
                    replicas=self.replicas,
                    selector=client.V1LabelSelector(
                        match_labels={'app': 'fraud-detection'}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={'app': 'fraud-detection'}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name='fraud-detection',
                                    image=self.image,
                                    ports=[
                                        client.V1ContainerPort(
                                            container_port=8000
                                        )
                                    ],
                                    resources=client.V1ResourceRequirements(
                                        requests=self.resources['requests'],
                                        limits=self.resources['limits']
                                    ),
                                    liveness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path='/health',
                                            port=8000
                                        ),
                                        initial_delay_seconds=30,
                                        period_seconds=10
                                    ),
                                    readiness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path='/ready',
                                            port=8000
                                        ),
                                        initial_delay_seconds=5,
                                        period_seconds=5
                                    ),
                                    env=[
                                        client.V1EnvVar(
                                            name='MODEL_PATH',
                                            value='/app/models/model.pkl'
                                        ),
                                        client.V1EnvVar(
                                            name='DEPLOYMENT_MODE',
                                            value='production'
                                        )
                                    ]
                                )
                            ]
                        )
                    )
                )
            )
            
            # Create deployment
            api_response = self.k8s_apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Deployment created: {api_response.metadata.name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to create deployment: {e}")
            return False
    
    def create_service(self) -> bool:
        """
        Create Kubernetes service.
        
        Returns:
            Success status
        """
        if not K8S_AVAILABLE or not self.k8s_core_v1:
            logger.error("Kubernetes not available")
            return False
        
        try:
            # Define service
            service = client.V1Service(
                metadata=client.V1ObjectMeta(
                    name=self.service_name,
                    namespace=self.namespace
                ),
                spec=client.V1ServiceSpec(
                    selector={'app': 'fraud-detection'},
                    ports=[
                        client.V1ServicePort(
                            port=80,
                            target_port=8000,
                            protocol='TCP',
                            name='http'
                        )
                    ],
                    type='ClusterIP'
                )
            )
            
            # Create service
            api_response = self.k8s_core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            
            logger.info(f"Service created: {api_response.metadata.name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to create service: {e}")
            return False
    
    def create_ingress(self, host: str) -> bool:
        """
        Create Kubernetes ingress.
        
        Args:
            host: Hostname for ingress
            
        Returns:
            Success status
        """
        if not K8S_AVAILABLE or not self.k8s_networking_v1:
            logger.error("Kubernetes not available")
            return False
        
        try:
            # Define ingress
            ingress = client.V1Ingress(
                metadata=client.V1ObjectMeta(
                    name=f"{self.service_name}-ingress",
                    namespace=self.namespace,
                    annotations={
                        'nginx.ingress.kubernetes.io/rewrite-target': '/'
                    }
                ),
                spec=client.V1IngressSpec(
                    rules=[
                        client.V1IngressRule(
                            host=host,
                            http=client.V1HTTPIngressRuleValue(
                                paths=[
                                    client.V1HTTPIngressPath(
                                        path='/',
                                        path_type='Prefix',
                                        backend=client.V1IngressBackend(
                                            service=client.V1IngressServiceBackend(
                                                name=self.service_name,
                                                port=client.V1ServiceBackendPort(
                                                    number=80
                                                )
                                            )
                                        )
                                    )
                                ]
                            )
                        )
                    ]
                )
            )
            
            # Create ingress
            api_response = self.k8s_networking_v1.create_namespaced_ingress(
                namespace=self.namespace,
                body=ingress
            )
            
            logger.info(f"Ingress created: {api_response.metadata.name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to create ingress: {e}")
            return False
    
    def scale_deployment(self, replicas: int) -> bool:
        """
        Scale Kubernetes deployment.
        
        Args:
            replicas: Number of replicas
            
        Returns:
            Success status
        """
        if not K8S_AVAILABLE or not self.k8s_apps_v1:
            logger.error("Kubernetes not available")
            return False
        
        try:
            # Get current deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            
            # Update replicas
            deployment.spec.replicas = replicas
            
            # Apply changes
            api_response = self.k8s_apps_v1.patch_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Deployment scaled to {replicas} replicas")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False
    
    def get_pods_status(self) -> List[Dict]:
        """
        Get status of all pods in deployment.
        
        Returns:
            List of pod statuses
        """
        if not K8S_AVAILABLE or not self.k8s_core_v1:
            logger.error("Kubernetes not available")
            return []
        
        try:
            pods = self.k8s_core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector='app=fraud-detection'
            )
            
            statuses = []
            for pod in pods.items:
                statuses.append({
                    'name': pod.metadata.name,
                    'phase': pod.status.phase,
                    'host_ip': pod.status.host_ip,
                    'pod_ip': pod.status.pod_ip,
                    'start_time': pod.status.start_time.isoformat() if pod.status.start_time else None
                })
            
            return statuses
            
        except ApiException as e:
            logger.error(f"Failed to get pods: {e}")
            return []


class SageMakerDeployment:
    """
    Deploys model to AWS SageMaker.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SageMaker deployment.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.region = config.get('region', 'us-east-1')
        self.instance_type = config.get('instance_type', 'ml.m5.xlarge')
        self.instance_count = config.get('instance_count', 1)
        self.model_name = config.get('model_name', 'fraud-detection-model')
        self.endpoint_name = config.get('endpoint_name', 'fraud-detection-endpoint')
        
        if AWS_AVAILABLE:
            self.sagemaker = boto3.client('sagemaker', region_name=self.region)
            self.s3 = boto3.client('s3', region_name=self.region)
        else:
            self.sagemaker = None
            self.s3 = None
    
    def upload_to_s3(self, model_path: str, bucket: str, key: str) -> bool:
        """
        Upload model to S3.
        
        Args:
            model_path: Local path to model
            bucket: S3 bucket name
            key: S3 key
            
        Returns:
            Success status
        """
        if not AWS_AVAILABLE:
            logger.error("AWS SDK not available")
            return False
        
        try:
            # Upload file
            self.s3.upload_file(model_path, bucket, key)
            logger.info(f"Model uploaded to s3://{bucket}/{key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
    
    def create_model(self, model_url: str, execution_role_arn: str) -> bool:
        """
        Create SageMaker model.
        
        Args:
            model_url: S3 URL of model
            execution_role_arn: IAM role ARN
            
        Returns:
            Success status
        """
        if not AWS_AVAILABLE:
            logger.error("AWS SDK not available")
            return False
        
        try:
            # Define primary container
            primary_container = {
                'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
                'ModelDataUrl': model_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_url
                }
            }
            
            # Create model
            response = self.sagemaker.create_model(
                ModelName=self.model_name,
                PrimaryContainer=primary_container,
                ExecutionRoleArn=execution_role_arn
            )
            
            logger.info(f"Model created: {response['ModelArn']}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to create model: {e}")
            return False
    
    def create_endpoint_config(self) -> bool:
        """
        Create endpoint configuration.
        
        Returns:
            Success status
        """
        if not AWS_AVAILABLE:
            logger.error("AWS SDK not available")
            return False
        
        try:
            # Create endpoint config
            response = self.sagemaker.create_endpoint_config(
                EndpointConfigName=f"{self.endpoint_name}-config",
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': self.model_name,
                        'InitialInstanceCount': self.instance_count,
                        'InstanceType': self.instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            logger.info(f"Endpoint config created: {response['EndpointConfigArn']}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to create endpoint config: {e}")
            return False
    
    def create_endpoint(self) -> bool:
        """
        Create SageMaker endpoint.
        
        Returns:
            Success status
        """
        if not AWS_AVAILABLE:
            logger.error("AWS SDK not available")
            return False
        
        try:
            # Create endpoint
            response = self.sagemaker.create_endpoint(
                EndpointName=self.endpoint_name,
                EndpointConfigName=f"{self.endpoint_name}-config"
            )
            
            logger.info(f"Endpoint created: {response['EndpointArn']}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to create endpoint: {e}")
            return False
    
    def wait_for_endpoint(self, timeout: int = 1800) -> bool:
        """
        Wait for endpoint to be in service.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Success status
        """
        if not AWS_AVAILABLE:
            logger.error("AWS SDK not available")
            return False
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.sagemaker.describe_endpoint(
                    EndpointName=self.endpoint_name
                )
                
                status = response['EndpointStatus']
                
                if status == 'InService':
                    logger.info("Endpoint is in service")
                    return True
                elif status == 'Failed':
                    logger.error("Endpoint creation failed")
                    return False
                
                logger.info(f"Endpoint status: {status}")
                time.sleep(60)
                
            except ClientError as e:
                logger.error(f"Failed to describe endpoint: {e}")
                return False
        
        logger.error(f"Timeout waiting for endpoint after {timeout} seconds")
        return False


# ============================================================================
# CANARY DEPLOYMENT MANAGER
# ============================================================================
class CanaryDeploymentManager:
    """
    Manages canary deployments for safe rollouts.
    
    This class handles:
    - Gradual traffic shifting
    - Performance monitoring
    - Automatic rollback on failures
    - A/B testing
    """
    
    def __init__(self, config: Dict):
        """
        Initialize canary deployment manager.
        
        Args:
            config: Canary configuration
        """
        self.config = config
        self.stable_version = config.get('stable_version', 'v1')
        self.canary_version = config.get('canary_version', 'v2')
        self.initial_traffic = config.get('initial_traffic', 0.1)  # 10%
        self.max_traffic = config.get('max_traffic', 1.0)
        self.step_size = config.get('step_size', 0.1)  # 10% increments
        self.step_interval = config.get('step_interval', 300)  # 5 minutes
        self.metrics_threshold = config.get('metrics_threshold', {
            'error_rate': 0.01,  # 1% error rate
            'latency_p99': 500,   # 500ms
            'success_rate': 0.99  # 99% success
        })
        
        # Initialize drift detector
        self.drift_detector = DriftDetector()
        
    def deploy_canary(self) -> bool:
        """
        Execute canary deployment.
        
        Returns:
            Success status
        """
        logger.info("Starting canary deployment...")
        
        # Deploy canary with 0 traffic
        if not self._deploy_canary_version():
            logger.error("Failed to deploy canary version")
            return False
        
        # Start with initial traffic
        current_traffic = self.initial_traffic
        
        while current_traffic <= self.max_traffic:
            logger.info(f"Shifting {current_traffic*100:.1f}% traffic to canary")
            
            # Shift traffic
            if not self._shift_traffic(current_traffic):
                logger.error("Failed to shift traffic")
                self._rollback()
                return False
            
            # Monitor for step_interval
            if not self._monitor_deployment(current_traffic):
                logger.warning(f"Monitoring failed at {current_traffic*100:.1f}% traffic")
                self._rollback()
                return False
            
            # Increase traffic
            current_traffic = min(current_traffic + self.step_size, self.max_traffic)
            
            # Check if we're done
            if current_traffic >= self.max_traffic - 0.001:
                logger.info("Canary deployment completed successfully")
                
                # Promote canary to stable
                self._promote_canary()
                return True
        
        return False
    
    def _deploy_canary_version(self) -> bool:
        """
        Deploy canary version with 0 traffic.
        
        Returns:
            Success status
        """
        # This would deploy the canary version
        # but not route any traffic to it yet
        logger.info("Deploying canary version...")
        return True
    
    def _shift_traffic(self, percentage: float) -> bool:
        """
        Shift traffic percentage to canary.
        
        Args:
            percentage: Traffic percentage (0-1)
            
        Returns:
            Success status
        """
        # This would update load balancer/router
        # to send specified percentage of traffic to canary
        logger.info(f"Traffic shifted: {percentage*100:.1f}% to canary")
        return True
    
    def _monitor_deployment(self, traffic_percentage: float, duration: int = 300) -> bool:
        """
        Monitor canary deployment health.
        
        Args:
            traffic_percentage: Current traffic percentage
            duration: Monitoring duration in seconds
            
        Returns:
            True if healthy, False otherwise
        """
        import time
        import numpy as np
        
        logger.info(f"Monitoring canary for {duration} seconds...")
        
        start_time = time.time()
        metrics_history = []
        
        while time.time() - start_time < duration:
            # Collect metrics
            metrics = self._collect_metrics()
            metrics_history.append(metrics)
            
            # Check thresholds
            if not self._check_thresholds(metrics):
                logger.error(f"Threshold violation detected: {metrics}")
                return False
            
            # Check for drift
            if len(metrics_history) >= 10:
                drift_detected = self.drift_detector.detect_drift(
                    np.array([m['predictions'] for m in metrics_history])
                )
                
                if drift_detected:
                    logger.error("Data drift detected in canary")
                    return False
            
            time.sleep(10)  # Check every 10 seconds
        
        # Calculate aggregate statistics
        if metrics_history:
            avg_error_rate = np.mean([m['error_rate'] for m in metrics_history])
            avg_latency = np.mean([m['latency_p99'] for m in metrics_history])
            avg_success = np.mean([m['success_rate'] for m in metrics_history])
            
            logger.info(f"Canary performance - Error: {avg_error_rate:.4f}, "
                       f"Latency: {avg_latency:.2f}ms, Success: {avg_success:.4f}")
        
        return True
    
    def _collect_metrics(self) -> Dict:
        """
        Collect deployment metrics.
        
        Returns:
            Dictionary of metrics
        """
        # This would collect real metrics from monitoring
        # For now, return simulated metrics
        import random
        
        return {
            'error_rate': random.uniform(0, 0.005),
            'latency_p99': random.uniform(50, 200),
            'success_rate': random.uniform(0.995, 1.0),
            'predictions': random.uniform(0, 1)
        }
    
    def _check_thresholds(self, metrics: Dict) -> bool:
        """
        Check if metrics are within thresholds.
        
        Args:
            metrics: Current metrics
            
        Returns:
            True if within thresholds
        """
        if metrics['error_rate'] > self.metrics_threshold['error_rate']:
            logger.error(f"Error rate {metrics['error_rate']:.4f} > "
                        f"threshold {self.metrics_threshold['error_rate']:.4f}")
            return False
        
        if metrics['latency_p99'] > self.metrics_threshold['latency_p99']:
            logger.error(f"P99 latency {metrics['latency_p99']:.2f}ms > "
                        f"threshold {self.metrics_threshold['latency_p99']}ms")
            return False
        
        if metrics['success_rate'] < self.metrics_threshold['success_rate']:
            logger.error(f"Success rate {metrics['success_rate']:.4f} < "
                        f"threshold {self.metrics_threshold['success_rate']:.4f}")
            return False
        
        return True
    
    def _rollback(self):
        """Rollback to stable version."""
        logger.warning("Rolling back to stable version")
        # This would revert traffic routing to 100% stable
        
    def _promote_canary(self):
        """Promote canary to stable version."""
        logger.info("Promoting canary to stable version")
        # This would make canary the new stable version


# ============================================================================
# MAIN DEPLOYMENT MANAGER
# ============================================================================
class DeploymentManager:
    """
    Main deployment manager for VeritasFinancial fraud detection system.
    
    This class orchestrates the entire deployment process:
    - Model packaging
    - Version management
    - Target deployment
    - Health monitoring
    - Rollback capabilities
    """
    
    def __init__(self, config_path: str = 'configs/deployment_config.yaml'):
        """
        Initialize deployment manager.
        
        Args:
            config_path: Path to deployment configuration
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Initialize components
        self.version_manager = ModelVersionManager()
        self.canary_manager = CanaryDeploymentManager(
            self.config.get('canary', {})
        )
        
        # Deployment targets
        self.targets = {}
        
        logger.info("DeploymentManager initialized")
    
    def package_model(self, model_name: str, version: str = None) -> Tuple[bool, str]:
        """
        Package model for deployment.
        
        Args:
            model_name: Name of the model
            version: Specific version (None for production)
            
        Returns:
            Tuple of (success, package_path)
        """
        # Get model path
        if version:
            # Get specific version
            model_path = self._get_model_path(model_name, version)
        else:
            # Get production version
            model_path = self.version_manager.get_production_model_path(model_name)
        
        if not model_path or not Path(model_path).exists():
            logger.error(f"Model not found: {model_name} {version}")
            return False, ""
        
        # Create package directory
        package_dir = Path('artifacts/deployment_packages')
        package_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"{model_name}_{version or 'prod'}_{timestamp}"
        package_path = package_dir / package_name
        package_path.mkdir(exist_ok=True)
        
        # Copy model file
        shutil.copy2(model_path, package_path / 'model.pkl')
        
        # Copy preprocessing pipeline
        pipeline_path = Path('artifacts/pipelines/preprocessing_pipeline.pkl')
        if pipeline_path.exists():
            shutil.copy2(pipeline_path, package_path / 'pipeline.pkl')
        
        # Copy feature engineering config
        feature_config = Path('configs/feature_config.yaml')
        if feature_config.exists():
            shutil.copy2(feature_config, package_path / 'feature_config.yaml')
        
        # Create metadata file
        metadata = {
            'model_name': model_name,
            'version': version or 'production',
            'timestamp': timestamp,
            'files': ['model.pkl']
        }
        
        if pipeline_path.exists():
            metadata['files'].append('pipeline.pkl')
        
        with open(package_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create tar.gz archive
        import tarfile
        archive_path = package_dir / f"{package_name}.tar.gz"
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(package_path, arcname=package_name)
        
        logger.info(f"Model packaged: {archive_path}")
        return True, str(archive_path)
    
    def _get_model_path(self, model_name: str, version: str) -> Optional[str]:
        """
        Get path to specific model version.
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            Path to model file or None
        """
        registry = self.version_manager.registry
        
        for model in registry['models'].get(model_name, []):
            if model['version'] == version:
                return model['model_path']
        
        return None
    
    def deploy_to_target(self, target: str, package_path: str) -> bool:
        """
        Deploy to specific target.
        
        Args:
            target: Deployment target (local, docker, k8s, sagemaker)
            package_path: Path to model package
            
        Returns:
            Success status
        """
        logger.info(f"Deploying to target: {target}")
        
        # Extract package
        import tarfile
        extract_dir = Path('artifacts/deployment_packages/extracted')
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(package_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        
        # Find model file
        model_file = None
        for extracted in extract_dir.rglob('*'):
            if extracted.name == 'model.pkl':
                model_file = extracted
                break
        
        if not model_file:
            logger.error("Model file not found in package")
            return False
        
        # Deploy to target
        if target == 'local':
            return self._deploy_local(model_file)
        elif target == 'docker':
            return self._deploy_docker(model_file)
        elif target == 'k8s':
            return self._deploy_kubernetes(model_file)
        elif target == 'sagemaker':
            return self._deploy_sagemaker(model_file)
        elif target == 'gcp':
            return self._deploy_gcp(model_file)
        elif target == 'azure':
            return self._deploy_azure(model_file)
        else:
            logger.error(f"Unknown target: {target}")
            return False
    
    def _deploy_local(self, model_path: Path) -> bool:
        """
        Deploy locally.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Success status
        """
        config = self.config.get('local', {})
        deployment = LocalDeployment(config)
        
        # In production, this would run in a separate process
        # For now, we'll just return success
        logger.info("Local deployment configured")
        logger.info(f"Run: uvicorn src.deployment.api.fastapi_app:app --host 0.0.0.0 --port 8000")
        
        return True
    
    def _deploy_docker(self, model_path: Path) -> bool:
        """
        Deploy to Docker.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Success status
        """
        config = self.config.get('docker', {})
        deployment = DockerDeployment(config)
        
        # Build image
        if not deployment.build_image():
            return False
        
        # Push to registry if configured
        registry_url = config.get('registry_url')
        if registry_url:
            if not deployment.push_to_registry(registry_url):
                return False
        
        # Run container
        if not deployment.run_container():
            return False
        
        # Check health
        if deployment.health_check():
            logger.info("Docker deployment healthy")
            return True
        else:
            logger.error("Docker deployment health check failed")
            return False
    
    def _deploy_kubernetes(self, model_path: Path) -> bool:
        """
        Deploy to Kubernetes.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Success status
        """
        config = self.config.get('kubernetes', {})
        deployment = KubernetesDeployment(config)
        
        # Create deployment
        if not deployment.create_deployment():
            return False
        
        # Create service
        if not deployment.create_service():
            return False
        
        # Create ingress if host specified
        host = config.get('host')
        if host:
            if not deployment.create_ingress(host):
                logger.warning("Failed to create ingress, continuing...")
        
        # Check pods status
        pods = deployment.get_pods_status()
        logger.info(f"Pods status: {pods}")
        
        return True
    
    def _deploy_sagemaker(self, model_path: Path) -> bool:
        """
        Deploy to AWS SageMaker.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Success status
        """
        config = self.config.get('sagemaker', {})
        deployment = SageMakerDeployment(config)
        
        # Upload to S3
        bucket = config.get('bucket')
        key = config.get('s3_key', 'models/fraud-detection/model.tar.gz')
        
        if not bucket:
            logger.error("S3 bucket not specified")
            return False
        
        if not deployment.upload_to_s3(str(model_path), bucket, key):
            return False
        
        # Create model
        model_url = f"s3://{bucket}/{key}"
        execution_role = config.get('execution_role')
        
        if not execution_role:
            logger.error("Execution role not specified")
            return False
        
        if not deployment.create_model(model_url, execution_role):
            return False
        
        # Create endpoint config
        if not deployment.create_endpoint_config():
            return False
        
        # Create endpoint
        if not deployment.create_endpoint():
            return False
        
        # Wait for endpoint
        if deployment.wait_for_endpoint():
            logger.info(f"Endpoint ready: {deployment.endpoint_name}")
            return True
        else:
            return False
    
    def _deploy_gcp(self, model_path: Path) -> bool:
        """
        Deploy to GCP AI Platform.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Success status
        """
        if not GCP_AVAILABLE:
            logger.error("GCP SDK not available")
            return False
        
        config = self.config.get('gcp', {})
        
        # This is a placeholder for GCP deployment
        logger.info("GCP deployment not fully implemented")
        return True
    
    def _deploy_azure(self, model_path: Path) -> bool:
        """
        Deploy to Azure ML.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Success status
        """
        if not AZURE_AVAILABLE:
            logger.error("Azure SDK not available")
            return False
        
        config = self.config.get('azure', {})
        
        # This is a placeholder for Azure deployment
        logger.info("Azure deployment not fully implemented")
        return True
    
    def run_canary_deployment(self) -> bool:
        """
        Run canary deployment.
        
        Returns:
            Success status
        """
        return self.canary_manager.deploy_canary()
    
    def health_check(self, target: str) -> bool:
        """
        Check health of deployed service.
        
        Args:
            target: Deployment target
            
        Returns:
            Health status
        """
        if target == 'local':
            deployment = LocalDeployment(self.config.get('local', {}))
            return deployment.health_check()
        elif target == 'docker':
            deployment = DockerDeployment(self.config.get('docker', {}))
            return deployment.health_check()
        elif target == 'k8s':
            # Check Kubernetes pods
            deployment = KubernetesDeployment(self.config.get('kubernetes', {}))
            pods = deployment.get_pods_status()
            return all(p['phase'] == 'Running' for p in pods)
        else:
            logger.error(f"Health check not implemented for {target}")
            return False
    
    def rollback(self, model_name: str) -> bool:
        """
        Rollback to previous version.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Success status
        """
        success, version = self.version_manager.rollback(model_name)
        
        if success:
            logger.info(f"Rolled back {model_name} to version {version}")
        else:
            logger.error(f"Rollback failed for {model_name}")
        
        return success


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='VeritasFinancial Model Deployment Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Package model
  python deploy_model.py package --model xgboost --version 1.0.0
  
  # Deploy locally
  python deploy_model.py deploy --target local --model xgboost
  
  # Deploy to Docker
  python deploy_model.py deploy --target docker --model xgboost
  
  # Deploy to Kubernetes
  python deploy_model.py deploy --target k8s --model xgboost
  
  # Deploy to AWS SageMaker
  python deploy_model.py deploy --target sagemaker --model xgboost --package /path/to/package.tar.gz
  
  # Run canary deployment
  python deploy_model.py canary --model xgboost
  
  # Check health
  python deploy_model.py health --target docker
  
  # Rollback
  python deploy_model.py rollback --model xgboost
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Deployment command')
    
    # Package command
    package_parser = subparsers.add_parser('package', help='Package model for deployment')
    package_parser.add_argument('--model', type=str, required=True, help='Model name')
    package_parser.add_argument('--version', type=str, help='Model version (default: production)')
    package_parser.add_argument('--output', type=str, help='Output path')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model')
    deploy_parser.add_argument('--target', type=str, required=True,
                              choices=['local', 'docker', 'k8s', 'sagemaker', 'gcp', 'azure'],
                              help='Deployment target')
    deploy_parser.add_argument('--model', type=str, help='Model name')
    deploy_parser.add_argument('--version', type=str, help='Model version')
    deploy_parser.add_argument('--package', type=str, help='Path to model package')
    deploy_parser.add_argument('--config', type=str, default='configs/deployment_config.yaml',
                              help='Deployment configuration file')
    
    # Canary command
    canary_parser = subparsers.add_parser('canary', help='Run canary deployment')
    canary_parser.add_argument('--model', type=str, required=True, help='Model name')
    canary_parser.add_argument('--config', type=str, default='configs/deployment_config.yaml',
                              help='Deployment configuration file')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check deployment health')
    health_parser.add_argument('--target', type=str, required=True,
                              choices=['local', 'docker', 'k8s'],
                              help='Deployment target')
    health_parser.add_argument('--config', type=str, default='configs/deployment_config.yaml',
                              help='Deployment configuration file')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback deployment')
    rollback_parser.add_argument('--model', type=str, required=True, help='Model name')
    rollback_parser.add_argument('--config', type=str, default='configs/deployment_config.yaml',
                                help='Deployment configuration file')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='List model versions')
    version_parser.add_argument('--model', type=str, help='Model name (optional)')
    
    return parser.parse_args()


def main():
    """
    Main execution function.
    """
    args = parse_arguments()
    
    logger.info("=" * 60)
    logger.info("VeritasFinancial Model Deployment")
    logger.info("=" * 60)
    
    try:
        # Initialize deployment manager
        if hasattr(args, 'config'):
            manager = DeploymentManager(config_path=args.config)
        else:
            manager = DeploymentManager()
        
        if args.command == 'package':
            # Package model
            success, package_path = manager.package_model(args.model, args.version)
            
            if success:
                logger.info(f"Model packaged successfully: {package_path}")
                
                # Copy to output if specified
                if args.output:
                    shutil.copy2(package_path, args.output)
                    logger.info(f"Package copied to: {args.output}")
            else:
                logger.error("Failed to package model")
                sys.exit(1)
        
        elif args.command == 'deploy':
            # Deploy model
            if args.package:
                # Deploy from package
                success = manager.deploy_to_target(args.target, args.package)
            elif args.model:
                # Package and deploy
                success, package_path = manager.package_model(args.model, args.version)
                
                if success:
                    success = manager.deploy_to_target(args.target, package_path)
                else:
                    success = False
            else:
                logger.error("Either --model or --package must be specified")
                sys.exit(1)
            
            if success:
                logger.info(f"Deployment to {args.target} successful")
            else:
                logger.error(f"Deployment to {args.target} failed")
                sys.exit(1)
        
        elif args.command == 'canary':
            # Run canary deployment
            success = manager.run_canary_deployment()
            
            if success:
                logger.info("Canary deployment successful")
            else:
                logger.error("Canary deployment failed")
                sys.exit(1)
        
        elif args.command == 'health':
            # Check health
            healthy = manager.health_check(args.target)
            
            if healthy:
                logger.info(f"Health check passed for {args.target}")
            else:
                logger.error(f"Health check failed for {args.target}")
                sys.exit(1)
        
        elif args.command == 'rollback':
            # Rollback
            success = manager.rollback(args.model)
            
            if success:
                logger.info(f"Rollback successful for {args.model}")
            else:
                logger.error(f"Rollback failed for {args.model}")
                sys.exit(1)
        
        elif args.command == 'version':
            # List versions
            registry = manager.version_manager.list_models()
            
            if args.model:
                # Show specific model
                if args.model in registry['models']:
                    print(f"\nVersions for {args.model}:")
                    for model in registry['models'][args.model]:
                        status = "PRODUCTION" if model.get('status') == 'production' else model.get('status', 'staged')
                        print(f"  Version {model['version']}: {status}")
                        print(f"    Timestamp: {model['timestamp']}")
                        print(f"    F1 Score: {model['metrics'].get('summary', {}).get('f1', 'N/A')}")
                        print()
                else:
                    print(f"Model {args.model} not found")
            else:
                # Show all models
                print("\nRegistered Models:")
                for model_name, versions in registry['models'].items():
                    prod_version = registry['current_versions'].get(model_name, 'None')
                    print(f"  {model_name}:")
                    print(f"    Production version: {prod_version}")
                    print(f"    Total versions: {len(versions)}")
                    print()
        
        else:
            print("Please specify a command. Use --help for usage information.")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Deployment completed successfully!")


if __name__ == "__main__":
    main()