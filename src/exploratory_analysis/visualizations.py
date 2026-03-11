"""
Visualization Module for Fraud Detection
=========================================
This module provides comprehensive visualization tools for exploring
fraud patterns in banking data. It includes:

1. Static Visualizations: Distribution plots, correlation heatmaps, etc.
2. Interactive Dashboards: Plotly-based interactive exploration
3. Geographic Mappings: Location-based fraud patterns
4. Time Series Visualizations: Temporal fraud trends
5. Custom Fraud Visualizations: Specialized plots for fraud analysis

All visualizations are designed to handle imbalanced data and
provide clear insights for fraud analysts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, PercentFormatter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning("folium not installed. Geographic visualizations will be limited.")

try:
    from ipywidgets import interact, widgets, Layout
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


class FraudVisualizer:
    """
    Main class for creating fraud detection visualizations.
    
    This class provides a unified interface for creating various
    types of visualizations for fraud analysis.
    
    Example:
        >>> visualizer = FraudVisualizer(df, target_col='is_fraud')
        >>> visualizer.create_overview_dashboard()
        >>> visualizer.save_report('fraud_analysis.html')
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = 'is_fraud',
        config: Optional[Dict] = None
    ):
        """
        Initialize the visualizer.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            config: Optional configuration dictionary
        """
        self.data = data.copy()
        self.target_col = target_col
        
        # Separate fraud and non-fraud data
        self.fraud_data = data[data[target_col] == 1]
        self.normal_data = data[data[target_col] == 0]
        
        # Set default configuration
        self.config = config or {}
        self.figsize = self.config.get('figsize', (12, 8))
        self.style = self.config.get('style', 'seaborn-v0_8-darkgrid')
        self.palette = self.config.get('palette', {
            'fraud': '#FF6B6B',
            'normal': '#4ECDC4',
            'fraud_light': '#FFA5A5',
            'normal_light': '#7FDFD6'
        })
        
        # Set style
        plt.style.use(self.style)
        
        logger.info(f"Initialized visualizer with {len(data)} transactions")
    
    def create_overview_dashboard(self) -> plt.Figure:
        """
        Create a comprehensive overview dashboard.
        
        This creates a multi-panel figure showing:
        - Class distribution
        - Amount distributions
        - Temporal patterns
        - Correlation heatmap
        
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating overview dashboard")
        
        # Create figure with GridSpec for complex layout
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Fraud Detection Overview Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Class Distribution (Pie Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_class_distribution(ax1)
        
        # 2. Amount Distribution (Box Plot)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_amount_distribution(ax2)
        
        # 3. Fraud Rate by Hour (Line Plot)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_fraud_by_hour(ax3)
        
        # 4. Fraud Rate by Day (Bar Plot)
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_fraud_by_day(ax4)
        
        # 5. Correlation Heatmap
        ax5 = fig.add_subplot(gs[1, :2])
        self._plot_correlation_heatmap(ax5)
        
        # 6. Amount Distribution (Histogram with KDE)
        ax6 = fig.add_subplot(gs[1, 2:])
        self._plot_amount_histogram(ax6)
        
        # 7. Fraud Amount vs Time (Scatter)
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_fraud_scatter(ax7)
        
        # 8. Feature Importance (Bar)
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_top_features(ax8)
        
        # 9. Missing Data Heatmap
        ax9 = fig.add_subplot(gs[2, 3])
        self._plot_missing_data(ax9)
        
        plt.tight_layout()
        return fig
    
    def _plot_class_distribution(self, ax: plt.Axes) -> None:
        """Plot class distribution as a pie chart."""
        class_counts = self.data[self.target_col].value_counts()
        labels = ['Normal', 'Fraud'] if 0 in class_counts.index else class_counts.index
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            class_counts.values,
            labels=labels,
            autopct='%1.2f%%',
            colors=[self.palette['normal'], self.palette['fraud']],
            explode=(0, 0.1),  # Slightly explode fraud slice
            shadow=True,
            startangle=90
        )
        
        # Customize text
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Class Distribution', fontweight='bold')
    
    def _plot_amount_distribution(self, ax: plt.Axes) -> None:
        """Plot amount distribution by class as box plot."""
        # Prepare data
        data_to_plot = [
            self.normal_data['Amount'].dropna(),
            self.fraud_data['Amount'].dropna()
        ]
        
        # Create box plot
        bp = ax.boxplot(
            data_to_plot,
            labels=['Normal', 'Fraud'],
            patch_artist=True,
            showfliers=False  # Hide outliers for better visibility
        )
        
        # Color boxes
        bp['boxes'][0].set_facecolor(self.palette['normal'])
        bp['boxes'][1].set_facecolor(self.palette['fraud'])
        
        ax.set_ylabel('Amount ($)')
        ax.set_title('Transaction Amount by Class', fontweight='bold')
        
        # Add grid for readability
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_fraud_by_hour(self, ax: plt.Axes) -> None:
        """Plot fraud transactions by hour of day."""
        if 'Time' in self.data.columns:
            # Convert time to hour
            self.data['Hour'] = (self.data['Time'] // 3600) % 24
            
            # Count fraud by hour
            fraud_by_hour = self.data[self.data[self.target_col] == 1].groupby('Hour').size()
            
            # Plot
            ax.plot(fraud_by_hour.index, fraud_by_hour.values, 
                   marker='o', color=self.palette['fraud'], linewidth=2)
            ax.fill_between(fraud_by_hour.index, fraud_by_hour.values, 
                           alpha=0.3, color=self.palette['fraud_light'])
            
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Number of Fraud Transactions')
            ax.set_title('Fraud Transactions by Hour', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 24, 2))
    
    def _plot_fraud_by_day(self, ax: plt.Axes) -> None:
        """Plot fraud rate by day of week."""
        if 'Time' in self.data.columns:
            # Convert time to day
            self.data['Day'] = (self.data['Time'] // (3600 * 24)) % 7
            
            # Calculate fraud rate by day
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fraud_rate = []
            
            for day in range(7):
                day_data = self.data[self.data['Day'] == day]
                if len(day_data) > 0:
                    rate = day_data[self.target_col].mean() * 100
                    fraud_rate.append(rate)
                else:
                    fraud_rate.append(0)
            
            # Plot
            bars = ax.bar(day_names, fraud_rate, color=self.palette['fraud'], alpha=0.7)
            
            # Add value labels on bars
            for bar, rate in zip(bars, fraud_rate):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.2f}%', ha='center', va='bottom')
            
            ax.set_ylabel('Fraud Rate (%)')
            ax.set_title('Fraud Rate by Day', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_correlation_heatmap(self, ax: plt.Axes) -> None:
        """Plot correlation heatmap of top features."""
        # Select numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        # Limit to top correlated features with target
        if len(numeric_cols) > 15:
            correlations = self.data[numeric_cols + [self.target_col]].corr()[self.target_col].abs()
            top_cols = correlations.nlargest(15).index.tolist()
            if self.target_col in top_cols:
                top_cols.remove(self.target_col)
            numeric_cols = top_cols[:10]  # Top 10 features
        
        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols + [self.target_col]].corr()
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlations', fontweight='bold')
    
    def _plot_amount_histogram(self, ax: plt.Axes) -> None:
        """Plot histogram of transaction amounts by class."""
        if 'Amount' in self.data.columns:
            # Use log scale for better visualization
            ax.hist(
                self.normal_data['Amount'].dropna(),
                bins=50,
                alpha=0.5,
                label='Normal',
                color=self.palette['normal'],
                density=True
            )
            ax.hist(
                self.fraud_data['Amount'].dropna(),
                bins=50,
                alpha=0.7,
                label='Fraud',
                color=self.palette['fraud'],
                density=True
            )
            
            ax.set_xscale('log')
            ax.set_xlabel('Amount ($) - Log Scale')
            ax.set_ylabel('Density')
            ax.set_title('Transaction Amount Distribution', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_fraud_scatter(self, ax: plt.Axes) -> None:
        """Create scatter plot of fraud transactions over time."""
        if len(self.fraud_data) > 0 and 'Time' in self.fraud_data.columns:
            # Sample if too many points
            plot_data = self.fraud_data
            if len(plot_data) > 1000:
                plot_data = plot_data.sample(n=1000, random_state=42)
            
            scatter = ax.scatter(
                plot_data['Time'] / 3600,  # Convert to hours
                plot_data['Amount'],
                alpha=0.6,
                c=plot_data['Amount'],
                cmap='YlOrRd',
                s=20
            )
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Amount ($)')
            ax.set_title('Fraud Transactions: Amount vs Time', fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Amount ($)')
            ax.grid(True, alpha=0.3)
    
    def _plot_top_features(self, ax: plt.Axes) -> None:
        """Plot top features by correlation with target."""
        # Calculate correlations with target
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        if numeric_cols:
            correlations = self.data[numeric_cols].corrwith(self.data[self.target_col]).abs()
            top_features = correlations.nlargest(5)
            
            # Create horizontal bar chart
            y_pos = range(len(top_features))
            ax.barh(y_pos, top_features.values, color=self.palette['fraud'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features.index)
            ax.set_xlabel('Absolute Correlation')
            ax.set_title('Top 5 Features by Correlation', fontweight='bold')
            ax.invert_yaxis()  # Display highest at top
            ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_missing_data(self, ax: plt.Axes) -> None:
        """Plot missing data heatmap."""
        missing_data = self.data.isnull()
        
        if missing_data.sum().sum() > 0:
            # Only show columns with missing data
            cols_with_missing = missing_data.columns[missing_data.any()].tolist()
            
            if cols_with_missing:
                # Sample rows if too many
                plot_data = missing_data[cols_with_missing]
                if len(plot_data) > 100:
                    plot_data = plot_data.sample(n=100, random_state=42)
                
                # Create heatmap
                sns.heatmap(
                    plot_data.T,
                    cmap=['#4ECDC4', '#FF6B6B'],  # Normal: teal, Missing: red
                    cbar=False,
                    xticklabels=False,
                    yticklabels=True,
                    ax=ax
                )
                
                ax.set_title('Missing Data Pattern\n(Red = Missing)', fontweight='bold')
                ax.set_xlabel('Sample Rows')
            else:
                ax.text(0.5, 0.5, 'No Missing Data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Missing Data', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Missing Data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Missing Data', fontweight='bold')
    
    def create_distribution_plots(
        self,
        columns: Optional[List[str]] = None,
        plot_type: str = 'histogram'
    ) -> plt.Figure:
        """
        Create distribution plots for specified columns.
        
        Args:
            columns: List of columns to plot (None for all numeric)
            plot_type: Type of plot ('histogram', 'kde', 'box', 'violin')
        
        Returns:
            matplotlib Figure object
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in columns if col != self.target_col]
        
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3  # 3 plots per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            if plot_type == 'histogram':
                self._plot_histogram(ax, col)
            elif plot_type == 'kde':
                self._plot_kde(ax, col)
            elif plot_type == 'box':
                self._plot_box(ax, col)
            elif plot_type == 'violin':
                self._plot_violin(ax, col)
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _plot_histogram(self, ax: plt.Axes, column: str) -> None:
        """Plot histogram for a single column."""
        ax.hist(
            self.normal_data[column].dropna(),
            bins=50,
            alpha=0.5,
            label='Normal',
            color=self.palette['normal'],
            density=True
        )
        ax.hist(
            self.fraud_data[column].dropna(),
            bins=50,
            alpha=0.7,
            label='Fraud',
            color=self.palette['fraud'],
            density=True
        )
        
        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        ax.set_title(f'{column} Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_kde(self, ax: plt.Axes, column: str) -> None:
        """Plot KDE for a single column."""
        if len(self.normal_data[column].dropna()) > 1:
            sns.kdeplot(
                data=self.normal_data,
                x=column,
                label='Normal',
                color=self.palette['normal'],
                fill=True,
                alpha=0.3,
                ax=ax
            )
        
        if len(self.fraud_data[column].dropna()) > 1:
            sns.kdeplot(
                data=self.fraud_data,
                x=column,
                label='Fraud',
                color=self.palette['fraud'],
                fill=True,
                alpha=0.5,
                ax=ax
            )
        
        ax.set_title(f'{column} Density by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_box(self, ax: plt.Axes, column: str) -> None:
        """Plot box plot for a single column."""
        data_to_plot = [
            self.normal_data[column].dropna(),
            self.fraud_data[column].dropna()
        ]
        
        bp = ax.boxplot(
            data_to_plot,
            labels=['Normal', 'Fraud'],
            patch_artist=True,
            showfliers=False
        )
        
        bp['boxes'][0].set_facecolor(self.palette['normal'])
        bp['boxes'][1].set_facecolor(self.palette['fraud'])
        
        ax.set_ylabel(column)
        ax.set_title(f'{column} Distribution by Class')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_violin(self, ax: plt.Axes, column: str) -> None:
        """Plot violin plot for a single column."""
        # Prepare data
        plot_data = pd.DataFrame({
            'value': pd.concat([
                self.normal_data[column],
                self.fraud_data[column]
            ]),
            'class': ['Normal'] * len(self.normal_data) + ['Fraud'] * len(self.fraud_data)
        })
        
        # Create violin plot
        sns.violinplot(
            data=plot_data,
            x='class',
            y='value',
            palette=[self.palette['normal'], self.palette['fraud']],
            ax=ax
        )
        
        ax.set_ylabel(column)
        ax.set_title(f'{column} Distribution by Class')
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_time_series_plots(
        self,
        time_col: str = 'Time',
        freq: str = 'H'
    ) -> plt.Figure:
        """
        Create time series visualizations.
        
        Args:
            time_col: Column containing time information
            freq: Resampling frequency ('H' for hourly, 'D' for daily, etc.)
        
        Returns:
            matplotlib Figure object
        """
        if time_col not in self.data.columns:
            logger.warning(f"Time column '{time_col}' not found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Convert time to datetime if needed
        if self.data[time_col].dtype in [np.int64, np.float64]:
            # Assuming time is in seconds
            self.data['datetime'] = pd.to_datetime(self.data[time_col], unit='s')
        else:
            self.data['datetime'] = pd.to_datetime(self.data[time_col])
        
        # Set datetime as index
        ts_data = self.data.set_index('datetime')
        
        # 1. Transaction Volume Over Time
        ax1 = axes[0, 0]
        volume = ts_data.resample(freq).size()
        ax1.plot(volume.index, volume.values, color='blue', linewidth=1)
        ax1.set_title('Transaction Volume')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Number of Transactions')
        ax1.grid(True, alpha=0.3)
        
        # 2. Fraud Rate Over Time
        ax2 = axes[0, 1]
        fraud_rate = ts_data[self.target_col].resample(freq).mean() * 100
        ax2.plot(fraud_rate.index, fraud_rate.values, color=self.palette['fraud'], linewidth=1)
        ax2.fill_between(fraud_rate.index, fraud_rate.values, alpha=0.3, color=self.palette['fraud_light'])
        ax2.set_title('Fraud Rate')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Fraud Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative Fraud
        ax3 = axes[1, 0]
        cumulative_fraud = ts_data[self.target_col].resample(freq).sum().cumsum()
        ax3.plot(cumulative_fraud.index, cumulative_fraud.values, color='darkred', linewidth=2)
        ax3.set_title('Cumulative Fraud Transactions')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Total Fraud')
        ax3.grid(True, alpha=0.3)
        
        # 4. Fraud Amount Over Time
        ax4 = axes[1, 1]
        if 'Amount' in self.data.columns:
            fraud_amount = ts_data[ts_data[self.target_col] == 1]['Amount'].resample(freq).sum()
            ax4.bar(fraud_amount.index, fraud_amount.values, color=self.palette['fraud'], alpha=0.7, width=0.8)
            ax4.set_title('Fraud Amount')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Total Amount ($)')
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Class Distribution', 'Amount by Class', 'Fraud by Hour',
                'Transaction Timeline', 'Fraud Rate by Day', 'Top Features',
                'Amount Distribution', 'Correlation Heatmap', 'Fraud Analysis'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'box'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'heatmap'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Class Distribution Pie Chart
        class_counts = self.data[self.target_col].value_counts()
        fig.add_trace(
            go.Pie(
                labels=['Normal', 'Fraud'],
                values=class_counts.values,
                marker=dict(colors=[self.palette['normal'], self.palette['fraud']]),
                textinfo='label+percent',
                hole=0.3
            ),
            row=1, col=1
        )
        
        # 2. Amount by Class Box Plot
        if 'Amount' in self.data.columns:
            fig.add_trace(
                go.Box(
                    y=self.normal_data['Amount'],
                    name='Normal',
                    marker_color=self.palette['normal'],
                    boxmean='sd'
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Box(
                    y=self.fraud_data['Amount'],
                    name='Fraud',
                    marker_color=self.palette['fraud'],
                    boxmean='sd'
                ),
                row=1, col=2
            )
        
        # 3. Fraud by Hour
        if 'Time' in self.data.columns:
            self.data['Hour'] = (self.data['Time'] // 3600) % 24
            fraud_by_hour = self.data[self.data[self.target_col] == 1].groupby('Hour').size()
            
            fig.add_trace(
                go.Scatter(
                    x=fraud_by_hour.index,
                    y=fraud_by_hour.values,
                    mode='lines+markers',
                    line=dict(color=self.palette['fraud'], width=2),
                    fill='tozeroy',
                    fillcolor=self.palette['fraud_light']
                ),
                row=1, col=3
            )
        
        # Update layout
        fig.update_layout(
            title_text="Fraud Detection Interactive Dashboard",
            showlegend=True,
            height=900,
            template='plotly_white'
        )
        
        return fig
    
    def save_report(self, filename: str = 'fraud_analysis_report.html') -> None:
        """
        Save comprehensive HTML report with all visualizations.
        
        Args:
            filename: Output filename
        """
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .section {{ margin: 30px 0; }}
                .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }}
                .stat-card {{ 
                    background: #f5f5f5; 
                    padding: 15px; 
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #FF6B6B; }}
                .stat-label {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Fraud Detection Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{len(self.data):,}</div>
                    <div class="stat-label">Total Transactions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(self.fraud_data):,}</div>
                    <div class="stat-label">Fraud Transactions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(self.fraud_data)/len(self.data)*100:.4f}%</div>
                    <div class="stat-label">Fraud Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.data.isnull().sum().sum():,}</div>
                    <div class="stat-label">Missing Values</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Interactive Dashboard</h2>
                <div id="dashboard"></div>
            </div>
            
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                // Add interactive dashboard here
                var dashboard = {self.create_interactive_dashboard().to_json()};
                Plotly.newPlot('dashboard', dashboard.data, dashboard.layout);
            </script>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {filename}")


class InteractiveDashboard:
    """
    Create interactive dashboards using ipywidgets.
    
    This class provides interactive widgets for exploring fraud data
    in Jupyter notebooks.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'is_fraud'):
        self.data = data
        self.target_col = target_col
        
        if not IPYWIDGETS_AVAILABLE:
            logger.warning("ipywidgets not installed. Interactive features disabled.")
    
    def create_explorer(self) -> Any:
        """
        Create an interactive data explorer.
        
        Returns:
            ipywidgets.VBox widget
        """
        if not IPYWIDGETS_AVAILABLE:
            return None
        
        # Create widgets
        feature_selector = widgets.SelectMultiple(
            options=self.data.columns.tolist(),
            value=[self.data.columns[0]],
            description='Features:',
            disabled=False,
            layout=Layout(width='50%', height='200px')
        )
        
        plot_type = widgets.Dropdown(
            options=['Histogram', 'Box Plot', 'Violin Plot', 'Scatter Plot'],
            value='Histogram',
            description='Plot Type:',
            disabled=False
        )
        
        output = widgets.Output()
        
        # Define update function
        def update_plot(change):
            with output:
                output.clear_output(wait=True)
                # Plot logic here
                print("Update plot with selected features")
        
        # Link widgets
        feature_selector.observe(update_plot, names='value')
        plot_type.observe(update_plot, names='value')
        
        # Create layout
        dashboard = widgets.VBox([
            widgets.HBox([feature_selector, plot_type]),
            output
        ])
        
        return dashboard


class DistributionPlotter:
    """Specialized class for distribution plotting."""
    
    @staticmethod
    def plot_qq(data: pd.Series, dist: str = 'norm') -> plt.Figure:
        """
        Create Q-Q plot to check distribution fit.
        
        Args:
            data: Data to plot
            dist: Distribution to compare against
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create Q-Q plot
        stats.probplot(data.dropna(), dist=dist, plot=ax)
        
        ax.set_title(f'Q-Q Plot against {dist} distribution')
        ax.grid(True, alpha=0.3)
        
        return fig


class TimeSeriesPlotter:
    """Specialized class for time series visualization."""
    
    @staticmethod
    def plot_seasonal_decomposition(
        data: pd.Series,
        period: int = 24,
        model: str = 'additive'
    ) -> plt.Figure:
        """
        Plot seasonal decomposition of time series.
        
        Args:
            data: Time series data
            period: Seasonal period
            model: Decomposition model ('additive' or 'multiplicative')
        
        Returns:
            matplotlib Figure
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Perform decomposition
        decomposition = seasonal_decompose(data, model=model, period=period)
        
        # Create plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original
        axes[0].plot(data.index, data.values)
        axes[0].set_title('Original Time Series')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(decomposition.trend.index, decomposition.trend.values)
        axes[1].set_title('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
        axes[2].set_title('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(decomposition.resid.index, decomposition.resid.values)
        axes[3].set_title('Residual')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class GeographicMapper:
    """Specialized class for geographic fraud visualization."""
    
    def __init__(self, data: pd.DataFrame, lat_col: str = 'lat', lon_col: str = 'lon'):
        self.data = data
        self.lat_col = lat_col
        self.lon_col = lon_col
        
        if not FOLIUM_AVAILABLE:
            logger.warning("folium not installed. Geographic mapping disabled.")
    
    def create_heatmap(self, center: Tuple[float, float] = (0, 0), zoom: int = 2) -> Any:
        """
        Create interactive heatmap of fraud locations.
        
        Args:
            center: Map center (lat, lon)
            zoom: Initial zoom level
        
        Returns:
            folium.Map object
        """
        if not FOLIUM_AVAILABLE:
            return None
        
        # Create base map
        m = folium.Map(location=center, zoom_start=zoom)
        
        # Prepare heatmap data
        heat_data = []
        for idx, row in self.data[self.data['is_fraud'] == 1].iterrows():
            if pd.notna(row[self.lat_col]) and pd.notna(row[self.lon_col]):
                heat_data.append([row[self.lat_col], row[self.lon_col]])
        
        # Add heatmap layer
        HeatMap(heat_data).add_to(m)
        
        return m
    
    def create_cluster_map(self, center: Tuple[float, float] = (0, 0), zoom: int = 2) -> Any:
        """
        Create clustered marker map.
        
        Args:
            center: Map center
            zoom: Initial zoom level
        
        Returns:
            folium.Map object
        """
        if not FOLIUM_AVAILABLE:
            return None
        
        m = folium.Map(location=center, zoom_start=zoom)
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in self.data.iterrows():
            if pd.notna(row[self.lat_col]) and pd.notna(row[self.lon_col]):
                # Choose color based on fraud status
                color = 'red' if row['is_fraud'] == 1 else 'blue'
                
                folium.Marker(
                    location=[row[self.lat_col], row[self.lon_col]],
                    popup=f"Transaction: {row.get('transaction_id', idx)}<br>"
                          f"Amount: ${row.get('Amount', 'N/A')}<br>"
                          f"Fraud: {row['is_fraud']}",
                    icon=folium.Icon(color=color)
                ).add_to(marker_cluster)
        
        return m