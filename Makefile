#!/usr/bin/make
# =============================================================================
# VERITASFINANCIAL - MAKEFILE
# =============================================================================
# Automation commands for development, testing, and deployment
# Usage: make [command]
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

# Shell to use
SHELL := /bin/bash

# Project name
PROJECT_NAME := veritasfinancial

# Python interpreter
PYTHON := python3

# Package manager
PIP := pip3

# Virtual environment directory
VENV_DIR := .venv

# Source directories
SRC_DIR := src
TEST_DIR := tests
NOTEBOOK_DIR := notebooks

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# =============================================================================
# HELP COMMAND (default target)
# =============================================================================
.PHONY: help
help:  ## Display this help message
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║     VERITASFINANCIAL - FRAUD DETECTION SYSTEM            ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "$(GREEN)%-25s %s$(NC)\n", "Command", "Description"}'
	@egrep -h '^[a-zA-Z_%-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make setup        # Set up development environment"
	@echo "  make train        # Train models"
	@echo "  make test         # Run tests"
	@echo ""

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

.PHONY: setup
setup: ## Set up complete development environment
	@echo "$(BLUE)🔧 Setting up VeritasFinancial development environment...$(NC)"
	@$(MAKE) venv
	@$(MAKE) install
	@$(MAKE) hooks
	@$(MAKE) directories
	@echo "$(GREEN)✅ Setup complete!$(NC)"
	@echo "$(YELLOW)👉 Activate virtual environment: source $(VENV_DIR)/bin/activate$(NC)"

.PHONY: venv
venv: ## Create virtual environment
	@echo "$(BLUE)📦 Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "$(GREEN)✅ Virtual environment created in $(VENV_DIR)$(NC)"

.PHONY: install
install: ## Install all dependencies
	@echo "$(BLUE)📚 Installing dependencies...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		$(PIP) install --upgrade pip setuptools wheel && \
		$(PIP) install -r requirements.txt && \
		$(PIP) install -r requirements-dev.txt && \
		$(PIP) install -e .
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

.PHONY: install-prod
install-prod: ## Install only production dependencies
	@echo "$(BLUE)📚 Installing production dependencies...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		$(PIP) install --upgrade pip && \
		$(PIP) install -r requirements.txt && \
		$(PIP) install -e .
	@echo "$(GREEN)✅ Production dependencies installed$(NC)"

.PHONY: install-gpu
install-gpu: ## Install with GPU support
	@echo "$(BLUE)🎮 Installing with GPU support...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
		$(PIP) install -r requirements.txt && \
		$(PIP) install -r requirements-dev.txt && \
		$(PIP) install -e .[gpu]
	@echo "$(GREEN)✅ GPU dependencies installed$(NC)"

.PHONY: hooks
hooks: ## Install pre-commit hooks
	@echo "$(BLUE)🔗 Installing pre-commit hooks...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		pre-commit install
	@echo "$(GREEN)✅ Pre-commit hooks installed$(NC)"

.PHONY: directories
directories: ## Create necessary directories
	@echo "$(BLUE)📁 Creating project directories...$(NC)"
	@mkdir -p data/{raw,processed,features,external}
	@mkdir -p models/{checkpoints,exports}
	@mkdir -p logs
	@mkdir -p cache
	@mkdir -p reports/{figures,html}
	@mkdir -p configs
	@echo "$(GREEN)✅ Directories created$(NC)"

# =============================================================================
# DATA PIPELINE
# =============================================================================

.PHONY: data
data: ## Download and prepare data
	@echo "$(BLUE)📥 Downloading data...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/download_data.py --config configs/data_config.yaml
	@echo "$(GREEN)✅ Data downloaded$(NC)"
	@$(MAKE) preprocess

.PHONY: preprocess
preprocess: ## Preprocess raw data
	@echo "$(BLUE)⚙️  Preprocessing data...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/preprocess_data.py --config configs/preprocessing_config.yaml
	@echo "$(GREEN)✅ Data preprocessed$(NC)"

.PHONY: features
features: ## Generate features
	@echo "$(BLUE)🔧 Generating features...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/generate_features.py --config configs/feature_config.yaml
	@echo "$(GREEN)✅ Features generated$(NC)"

.PHONY: validate-data
validate-data: ## Validate data quality
	@echo "$(BLUE)🔍 Validating data quality...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/validate_data.py --config configs/validation_config.yaml
	@echo "$(GREEN)✅ Data validation complete$(NC)"

# =============================================================================
# EXPLORATORY ANALYSIS
# =============================================================================

.PHONY: eda
eda: ## Run exploratory data analysis
	@echo "$(BLUE)📊 Running exploratory data analysis...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		jupyter nbconvert --execute --to notebook --inplace \
			notebooks/01_eda_statistical_analysis.ipynb
	@echo "$(GREEN)✅ EDA complete - Check notebooks/01_eda_statistical_analysis.ipynb$(NC)"

.PHONY: visualize
visualize: ## Generate visualizations
	@echo "$(BLUE)📈 Generating visualizations...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/generate_visualizations.py --output-dir reports/figures/
	@echo "$(GREEN)✅ Visualizations saved to reports/figures/$(NC)"

# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

.PHONY: train
train: ## Train all models
	@echo "$(BLUE)🤖 Training models...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/train_model.py --config configs/model_config.yaml
	@echo "$(GREEN)✅ Model training complete$(NC)"

.PHONY: train-xgboost
train-xgboost: ## Train XGBoost model only
	@echo "$(BLUE)🤖 Training XGBoost model...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/train_model.py --model xgboost --config configs/model_config.yaml
	@echo "$(GREEN)✅ XGBoost training complete$(NC)"

.PHONY: train-lightgbm
train-lightgbm: ## Train LightGBM model only
	@echo "$(BLUE)🤖 Training LightGBM model...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/train_model.py --model lightgbm --config configs/model_config.yaml
	@echo "$(GREEN)✅ LightGBM training complete$(NC)"

.PHONY: train-transformer
train-transformer: ## Train Transformer model only
	@echo "$(BLUE)🤖 Training Transformer model...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/train_model.py --model transformer --config configs/model_config.yaml
	@echo "$(GREEN)✅ Transformer training complete$(NC)"

.PHONY: evaluate
evaluate: ## Evaluate models
	@echo "$(BLUE)📏 Evaluating models...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/evaluate_model.py --config configs/evaluation_config.yaml
	@echo "$(GREEN)✅ Evaluation complete$(NC)"

.PHONY: backtest
backtest: ## Run backtesting
	@echo "$(BLUE)🔄 Running backtesting...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/backtest.py --config configs/backtest_config.yaml
	@echo "$(GREEN)✅ Backtesting complete$(NC)"

.PHONY: optimize
optimize: ## Hyperparameter optimization
	@echo "$(BLUE)🎯 Running hyperparameter optimization...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/optimize_hyperparameters.py --config configs/optimization_config.yaml
	@echo "$(GREEN)✅ Hyperparameter optimization complete$(NC)"

# =============================================================================
# TESTING
# =============================================================================

.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term --cov-report=html
	@echo "$(GREEN)✅ Tests complete - Coverage report in htmlcov/$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(BLUE)🧪 Running unit tests...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		pytest $(TEST_DIR)/unit -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(BLUE)🧪 Running integration tests...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		pytest $(TEST_DIR)/integration -v

.PHONY: test-slow
test-slow: ## Run slow tests
	@echo "$(BLUE)🐢 Running slow tests...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		pytest $(TEST_DIR) -v -m slow

.PHONY: coverage
coverage: ## Generate coverage report
	@echo "$(BLUE)📊 Generating coverage report...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=xml
	@echo "$(GREEN)✅ Coverage reports in htmlcov/ and coverage.xml$(NC)"

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: lint
lint: ## Run linters
	@echo "$(BLUE)🔍 Running linters...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		ruff check $(SRC_DIR) $(TEST_DIR) && \
		black --check $(SRC_DIR) $(TEST_DIR) && \
		mypy $(SRC_DIR)
	@echo "$(GREEN)✅ Linting complete$(NC)"

.PHONY: format
format: ## Format code
	@echo "$(BLUE)✨ Formatting code...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		black $(SRC_DIR) $(TEST_DIR) && \
		ruff check $(SRC_DIR) $(TEST_DIR) --fix
	@echo "$(GREEN)✅ Code formatted$(NC)"

.PHONY: typecheck
typecheck: ## Run type checker
	@echo "$(BLUE)🔤 Running type checker...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		mypy $(SRC_DIR) --strict
	@echo "$(GREEN)✅ Type checking complete$(NC)"

.PHONY: security
security: ## Run security checks
	@echo "$(BLUE)🛡️  Running security checks...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		bandit -r $(SRC_DIR) -f json -o security_report.json && \
		safety check -r requirements.txt
	@echo "$(GREEN)✅ Security checks complete$(NC)"

# =============================================================================
# DEPLOYMENT
# =============================================================================

.PHONY: serve
serve: ## Start API server locally
	@echo "$(BLUE)🚀 Starting API server...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		uvicorn src.deployment.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload

.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	@docker build -t $(PROJECT_NAME):latest .
	@docker build -t $(PROJECT_NAME):$(shell git rev-parse --short HEAD) .
	@echo "$(GREEN)✅ Docker image built$(NC)"

.PHONY: docker-run
docker-run: ## Run Docker container
	@echo "$(BLUE)🐳 Running Docker container...$(NC)"
	@docker run -p 8000:8000 -v $(PWD)/models:/app/models $(PROJECT_NAME):latest

.PHONY: docker-compose-up
docker-compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)🐳 Starting services...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✅ Services started$(NC)"

.PHONY: docker-compose-down
docker-compose-down: ## Stop all services
	@echo "$(BLUE)🐳 Stopping services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✅ Services stopped$(NC)"

.PHONY: deploy-aws
deploy-aws: ## Deploy to AWS
	@echo "$(BLUE)☁️  Deploying to AWS...$(NC)"
	@cd deployment/cloud/aws && ./scripts/deploy.sh
	@echo "$(GREEN)✅ AWS deployment complete$(NC)"

.PHONY: deploy-gcp
deploy-gcp: ## Deploy to GCP
	@echo "$(BLUE)☁️  Deploying to GCP...$(NC)"
	@cd deployment/cloud/gcp && ./scripts/deploy.sh
	@echo "$(GREEN)✅ GCP deployment complete$(NC)"

.PHONY: deploy-azure
deploy-azure: ## Deploy to Azure
	@echo "$(BLUE)☁️  Deploying to Azure...$(NC)"
	@cd deployment/cloud/azure && ./scripts/deploy.sh
	@echo "$(GREEN)✅ Azure deployment complete$(NC)"

# =============================================================================
# DOCUMENTATION
# =============================================================================

.PHONY: docs
docs: ## Generate documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		mkdocs build
	@echo "$(GREEN)✅ Documentation generated in site/$(NC)"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "$(BLUE)📚 Serving documentation...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		mkdocs serve

.PHONY: api-docs
api-docs: ## Generate API documentation
	@echo "$(BLUE)📚 Generating API documentation...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		pdoc src/$(PROJECT_NAME) -o docs/api
	@echo "$(GREEN)✅ API docs generated in docs/api/$(NC)"

# =============================================================================
# MONITORING & PROFILING
# =============================================================================

.PHONY: monitor
monitor: ## Start monitoring dashboard
	@echo "$(BLUE)📊 Starting monitoring dashboard...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python scripts/monitor.py --dashboard

.PHONY: profile
profile: ## Run profiling
	@echo "$(BLUE)⏱️  Running profiling...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		python -m cProfile -o profile.stats scripts/train_model.py && \
		python -m pstats profile.stats
	@echo "$(GREEN)✅ Profiling complete$(NC)"

# =============================================================================
# CLEANUP
# =============================================================================

.PHONY: clean
clean: ## Clean up temporary files
	@echo "$(BLUE)🧹 Cleaning up...$(NC)"
	# Python cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	
	# Python files
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	
	# Build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	
	# Logs
	rm -rf logs/*
	
	# Jupyter checkpoints
	rm -rf .ipynb_checkpoints/
	
	# Profile data
	rm -f profile.stats
	
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

.PHONY: clean-all
clean-all: clean ## Remove everything including virtual environment and data
	@echo "$(BLUE)🧹 Deep cleaning...$(NC)"
	rm -rf $(VENV_DIR)/
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf cache/*
	@echo "$(GREEN)✅ Deep clean complete$(NC)"

.PHONY: clean-docker
clean-docker: ## Clean Docker resources
	@echo "$(BLUE)🐳 Cleaning Docker...$(NC)"
	docker system prune -f
	docker volume prune -f
	@echo "$(GREEN)✅ Docker clean complete$(NC)"

# =============================================================================
# VERSION MANAGEMENT
# =============================================================================

.PHONY: version
version: ## Show current version
	@echo "$(BLUE)📌 Current version: $(GREEN)$(shell cat VERSION)$(NC)"

.PHONY: bump-patch
bump-patch: ## Bump patch version (1.0.0 -> 1.0.1)
	@echo "$(BLUE)🔖 Bumping patch version...$(NC)"
	@$(eval OLD_VERSION=$(shell cat VERSION))
	@$(eval NEW_VERSION=$(shell echo $(OLD_VERSION) | awk -F. '{$$3+=1; print $$1"."$$2"."$$3}'))
	@echo $(NEW_VERSION) > VERSION
	@echo "$(GREEN)✅ Version bumped: $(OLD_VERSION) -> $(NEW_VERSION)$(NC)"

.PHONY: bump-minor
bump-minor: ## Bump minor version (1.0.0 -> 1.1.0)
	@echo "$(BLUE)🔖 Bumping minor version...$(NC)"
	@$(eval OLD_VERSION=$(shell cat VERSION))
	@$(eval NEW_VERSION=$(shell echo $(OLD_VERSION) | awk -F. '{$$2+=1; $$3=0; print $$1"."$$2"."$$3}'))
	@echo $(NEW_VERSION) > VERSION
	@echo "$(GREEN)✅ Version bumped: $(OLD_VERSION) -> $(NEW_VERSION)$(NC)"

.PHONY: bump-major
bump-major: ## Bump major version (1.0.0 -> 2.0.0)
	@echo "$(BLUE)🔖 Bumping major version...$(NC)"
	@$(eval OLD_VERSION=$(shell cat VERSION))
	@$(eval NEW_VERSION=$(shell echo $(OLD_VERSION) | awk -F. '{$$1+=1; $$2=0; $$3=0; print $$1"."$$2"."$$3}'))
	@echo $(NEW_VERSION) > VERSION
	@echo "$(GREEN)✅ Version bumped: $(OLD_VERSION) -> $(NEW_VERSION)$(NC)"

# =============================================================================
# UTILITIES
# =============================================================================

.PHONY: shell
shell: ## Open Python shell with project context
	@echo "$(BLUE)🐍 Opening Python shell...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		PYTHONPATH=$(PWD)/src python

.PHONY: notebook
notebook: ## Start Jupyter notebook
	@echo "$(BLUE)📓 Starting Jupyter notebook...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		jupyter notebook --notebook-dir=$(NOTEBOOK_DIR)

.PHONY: lab
lab: ## Start JupyterLab
	@echo "$(BLUE)📓 Starting JupyterLab...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		jupyter lab --notebook-dir=$(NOTEBOOK_DIR)

.PHONY: tree
tree: ## Show project tree structure
	@echo "$(BLUE)🌳 Project structure:$(NC)"
	@tree -I '$(VENV_DIR)|__pycache__|*.pyc|.git' --dirsfirst

# =============================================================================
# END OF MAKEFILE
# =============================================================================