# =============================================================================
# VERITASFINANCIAL - SETUP.PY
# =============================================================================
"""
Setup script for VeritasFinancial Fraud Detection System.

This file is maintained for backward compatibility with older Python versions.
Modern builds use pyproject.toml, but setup.py is still needed for:
- Editable installs (pip install -e .)
- Some legacy build systems
- Certain development workflows

This setup.py reads configuration from pyproject.toml to maintain
a single source of truth.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def read_file(filename: str) -> str:
    """
    Read and return the contents of a file.
    
    Args:
        filename: Name of the file to read
        
    Returns:
        String content of the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    file_path = Path(__file__).parent / filename
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_requirements() -> list:
    """
    Parse requirements from requirements.txt file.
    
    Returns:
        List of package requirements
    """
    requirements = []
    req_file = Path(__file__).parent / 'requirements.txt'
    
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and editable installs
                if line and not line.startswith('#') and not line.startswith('-e'):
                    requirements.append(line)
    
    return requirements


def get_version() -> str:
    """
    Get the package version.
    Priority: environment variable > version file > default
    
    Returns:
        Version string
    """
    # Check environment variable first (for CI/CD)
    env_version = os.environ.get('VERITAS_VERSION')
    if env_version:
        return env_version
    
    # Try to read from VERSION file
    version_file = Path(__file__).parent / 'VERSION'
    if version_file.exists():
        return read_file('VERSION')
    
    # Default version
    return '1.0.0'


def get_long_description() -> str:
    """
    Get the long description from README.md.
    
    Returns:
        README content as string
    """
    try:
        return read_file('README.md')
    except FileNotFoundError:
        return "VeritasFinancial - Banking Fraud Detection System"


# =============================================================================
# MAIN SETUP CONFIGURATION
# =============================================================================
setup(
    # =========================================================================
    # PACKAGE METADATA
    # =========================================================================
    name='veritasfinancial',
    version=get_version(),
    description='Enterprise Banking Fraud Detection System using Classical ML, Transformers, and LLMs',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    
    # Author information
    author='Veritas Financial Data Science Team',
    author_email='datascience@veritasfinancial.com',
    maintainer='MLOps Team',
    maintainer_email='mlops@veritasfinancial.com',
    
    # URLs
    url='https://github.com/veritasfinancial/fraud-detection',
    download_url='https://github.com/veritasfinancial/fraud-detection/releases',
    project_urls={
        'Documentation': 'https://docs.veritasfinancial.com/fraud-detection',
        'Source': 'https://github.com/veritasfinancial/fraud-detection',
        'Tracker': 'https://github.com/veritasfinancial/fraud-detection/issues',
        'Changelog': 'https://github.com/veritasfinancial/fraud-detection/blob/main/CHANGELOG.md',
    },
    
    # =========================================================================
    # PACKAGE CLASSIFICATION
    # =========================================================================
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Security',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    # =========================================================================
    # PACKAGE DISCOVERY
    # =========================================================================
    # Find all packages in src directory
    packages=find_packages(
        where='src',                    # Look in src directory
        include=['veritasfinancial*'],   # Include all veritasfinancial packages
        exclude=['tests*', 'scripts*'],  # Exclude test directories
    ),
    package_dir={'': 'src'},            # Root package is in src directory
    
    # Include package data files
    include_package_data=True,
    
    # =========================================================================
    # DEPENDENCIES
    # =========================================================================
    # Production dependencies
    install_requires=get_requirements(),
    
    # Python version requirement
    python_requires='>=3.10, <3.13',
    
    # =========================================================================
    # OPTIONAL DEPENDENCIES
    # =========================================================================
    extras_require={
        # GPU support
        'gpu': [
            'torch>=2.0.0+cu118',
            'xformers>=0.0.20',
            'triton>=2.0.0',
        ],
        
        # Development tools
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'ruff>=0.0.280',
            'mypy>=1.4.0',
            'pre-commit>=3.3.0',
            'jupyter>=1.0.0',
            'ipykernel>=6.25.0',
            'ipywidgets>=8.0.0',
        ],
        
        # Monitoring and observability
        'monitoring': [
            'prometheus-client>=0.17.0',
            'sentry-sdk>=1.30.0',
            'opentelemetry-api>=1.19.0',
            'opentelemetry-sdk>=1.19.0',
            'opentelemetry-instrumentation-fastapi>=0.40b0',
        ],
        
        # Cloud deployment
        'cloud': [
            'boto3>=1.28.0',                    # AWS
            'google-cloud-storage>=2.10.0',      # GCP
            'azure-storage-blob>=12.17.0',       # Azure
            'kubernetes>=26.0.0',                 # K8s
        ],
        
        # Visualization extras
        'viz': [
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'plotly>=5.14.0',
            'bokeh>=3.1.0',
            'yellowbrick>=1.5.0',
        ],
        
        # Documentation
        'docs': [
            'mkdocs>=1.5.0',
            'mkdocs-material>=9.1.0',
            'pdoc>=13.0.0',
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
        
        # All optional dependencies
        'all': [
            'veritasfinancial[gpu,dev,monitoring,cloud,viz,docs]',
        ],
    },
    
    # =========================================================================
    # ENTRY POINTS (Command Line Scripts)
    # =========================================================================
    entry_points={
        'console_scripts': [
            # Main CLI commands
            'fraud-train = veritasfinancial.cli.train:main',
            'fraud-predict = veritasfinancial.cli.predict:main',
            'fraud-evaluate = veritasfinancial.cli.evaluate:main',
            'fraud-serve = veritasfinancial.cli.serve:main',
            'fraud-analyze = veritasfinancial.cli.analyze:main',
            
            # Utility scripts
            'fraud-features = veritasfinancial.cli.features:main',
            'fraud-monitor = veritasfinancial.cli.monitor:main',
            'fraud-backtest = veritasfinancial.cli.backtest:main',
            'fraud-explain = veritasfinancial.cli.explain:main',
            
            # Data pipeline scripts
            'fraud-ingest = veritasfinancial.cli.ingest:main',
            'fraud-preprocess = veritasfinancial.cli.preprocess:main',
            'fraud-validate = veritasfinancial.cli.validate:main',
        ],
        
        # GUI applications (if any)
        'gui_scripts': [
            'fraud-dashboard = veritasfinancial.visualization.dashboard:run',
        ],
        
        # Plugin system (for extensibility)
        'veritasfinancial.plugins': [
            'xgboost = veritasfinancial.plugins.xgboost_plugin',
            'lightgbm = veritasfinancial.plugins.lightgbm_plugin',
            'transformer = veritasfinancial.plugins.transformer_plugin',
        ],
    },
    
    # =========================================================================
    # PACKAGE DATA FILES
    # =========================================================================
    package_data={
        'veritasfinancial': [
            'configs/*.yaml',
            'configs/*.json',
            'models/*.pkl',
            'data/schemas/*.json',
            'static/css/*.css',
            'static/js/*.js',
            'templates/*.html',
        ],
    },
    
    # Data files outside the package
    data_files=[
        ('share/veritasfinancial', [
            'README.md',
            'LICENSE',
            'CHANGELOG.md',
        ]),
        ('share/veritasfinancial/configs', [
            'configs/default_config.yaml',
            'configs/production_config.yaml',
        ]),
    ],
    
    # =========================================================================
    # BUILD AND INSTALLATION OPTIONS
    # =========================================================================
    zip_safe=False,                     # Don't use zip for package
    platforms='any',                     # Platform independent
    
    # =========================================================================
    # METADATA FOR PYPI
    # =========================================================================
    license='Proprietary',
    keywords='fraud-detection banking machine-learning deep-learning transformers llm',
    
    # =========================================================================
    # CUSTOM COMMANDS
    # =========================================================================
    cmdclass={
        # Custom setup commands can be added here
        # 'test': PyTest,
        # 'coverage': Coverage,
    },
)


# =============================================================================
# POST-INSTALLATION HOOK
# =============================================================================
def post_install():
    """
    Run after installation to set up directories, download models, etc.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create necessary directories
        dirs = [
            'data/raw',
            'data/processed',
            'data/features',
            'models',
            'logs',
            'cache',
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Download default models if needed
        # download_default_models()
        
        logger.info("VeritasFinancial installation completed successfully!")
        
    except Exception as e:
        logger.error(f"Post-installation setup failed: {e}")
        logger.warning("You may need to set up directories manually.")


# Run post-install if not in a build environment
if not any(arg in sys.argv for arg in ['sdist', 'bdist_wheel', 'egg_info']):
    post_install()

# =============================================================================
# END OF setup.py
# =============================================================================