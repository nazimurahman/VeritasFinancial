"""
API Endpoints
=============

Defines all API endpoints for the fraud detection system.
Includes prediction, explanation, and management endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Security scheme for authentication
security = HTTPBearer(auto_error=False)

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class TransactionData(BaseModel):
    """
    Transaction data model for fraud detection requests.
    
    This model defines the structure and validation rules for incoming
    transaction data that needs to be checked for fraud.
    """
    
    transaction_id: Optional[str] = Field(
        None,
        description="Unique transaction identifier (auto-generated if not provided)"
    )
    customer_id: str = Field(
        ...,
        description="Unique customer identifier",
        min_length=5,
        max_length=50
    )
    amount: float = Field(
        ...,
        description="Transaction amount in the specified currency",
        gt=0,  # Greater than 0
        le=1000000  # Less than or equal to 1 million
    )
    currency: str = Field(
        ...,
        description="Transaction currency code (ISO 4217)",
        min_length=3,
        max_length=3
    )
    merchant_id: str = Field(
        ...,
        description="Merchant identifier",
        min_length=3,
        max_length=50
    )
    merchant_category: Optional[str] = Field(
        None,
        description="Merchant category code"
    )
    transaction_time: datetime = Field(
        ...,
        description="Transaction timestamp (ISO 8601 format)"
    )
    device_id: Optional[str] = Field(
        None,
        description="Device identifier for the transaction"
    )
    ip_address: Optional[str] = Field(
        None,
        description="IP address of the transaction origin",
        regex=r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"  # Simple IPv4 validation
    )
    country: Optional[str] = Field(
        None,
        description="Country code (ISO 3166-1 alpha-2)",
        min_length=2,
        max_length=2
    )
    city: Optional[str] = Field(
        None,
        description="City name"
    )
    
    @validator('currency')
    def validate_currency(cls, v):
        """Validate currency code against allowed list."""
        allowed_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY'}
        if v.upper() not in allowed_currencies:
            raise ValueError(f'Currency must be one of: {allowed_currencies}')
        return v.upper()
    
    @validator('transaction_time')
    def validate_transaction_time(cls, v):
        """Ensure transaction time is not in the future."""
        if v > datetime.utcnow():
            raise ValueError('Transaction time cannot be in the future')
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "customer_id": "cust_123456",
                "amount": 1250.50,
                "currency": "USD",
                "merchant_id": "merch_789",
                "transaction_time": "2024-01-15T14:30:00Z",
                "device_id": "dev_abc123",
                "ip_address": "192.168.1.100",
                "country": "US",
                "city": "New York"
            }
        }

class FraudPredictionResponse(BaseModel):
    """
    Fraud prediction response model.
    
    Contains the fraud probability score and additional information
    about the prediction.
    """
    
    transaction_id: str = Field(
        ...,
        description="Transaction identifier"
    )
    fraud_probability: float = Field(
        ...,
        description="Probability of fraud (0-1)",
        ge=0,
        le=1
    )
    risk_level: str = Field(
        ...,
        description="Risk level classification",
        regex="^(low|medium|high|critical)$"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(
        ...,
        description="Response timestamp"
    )
    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_123456",
                "fraud_probability": 0.87,
                "risk_level": "high",
                "model_version": "xgboost_v2.1.0",
                "processing_time_ms": 45.2,
                "timestamp": "2024-01-15T14:30:05.123Z",
                "request_id": "req_abc123"
            }
        }

class ExplainabilityResponse(BaseModel):
    """
    Explainability response model.
    
    Provides explanations for why a transaction was flagged as potentially fraudulent.
    """
    
    transaction_id: str = Field(..., description="Transaction identifier")
    fraud_probability: float = Field(..., description="Fraud probability")
    top_risk_factors: List[Dict[str, Any]] = Field(
        ...,
        description="Top factors contributing to the risk score"
    )
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Feature importance values for this prediction"
    )
    shap_values: Optional[Dict[str, float]] = Field(
        None,
        description="SHAP values for explainability"
    )
    rule_triggers: List[str] = Field(
        default_factory=list,
        description="Business rules that were triggered"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_123456",
                "fraud_probability": 0.87,
                "top_risk_factors": [
                    {
                        "factor": "unusual_amount",
                        "impact": 0.45,
                        "description": "Amount is 3x higher than average"
                    },
                    {
                        "factor": "new_device",
                        "impact": 0.30,
                        "description": "Transaction from unrecognized device"
                    }
                ],
                "feature_importance": {
                    "amount_ratio": 0.45,
                    "device_risk": 0.30,
                    "velocity": 0.25
                }
            }
        }

class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request model.
    
    For processing multiple transactions in a single request.
    """
    
    transactions: List[TransactionData] = Field(
        ...,
        description="List of transactions to evaluate",
        max_items=1000
    )
    async_processing: bool = Field(
        False,
        description="Whether to process asynchronously"
    )
    webhook_url: Optional[str] = Field(
        None,
        description="Webhook URL for async completion notification"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "transactions": [
                    {"customer_id": "cust_1", "amount": 100, ...},
                    {"customer_id": "cust_2", "amount": 250, ...}
                ],
                "async_processing": False
            }
        }

class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response model.
    """
    
    batch_id: str = Field(..., description="Batch identifier")
    total_transactions: int = Field(..., description="Number of transactions")
    results: List[FraudPredictionResponse] = Field(
        ...,
        description="Prediction results"
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "batch_id": "batch_123456",
                "total_transactions": 10,
                "results": [],
                "summary": {
                    "avg_risk_score": 0.25,
                    "high_risk_count": 2,
                    "processing_time_seconds": 1.2
                }
            }
        }

# ============================================================================
# Authentication Dependency
# ============================================================================

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Verify API token for authentication.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        str: Customer ID or API key
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    
    # In production, validate token against database or auth service
    # This is a simplified example
    if not token.startswith("vf_"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication token"
        )
    
    # Extract customer ID from token (simplified)
    customer_id = token[3:]  # Remove "vf_" prefix
    
    logger.info(f"Authenticated customer: {customer_id}")
    return customer_id

# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/predict",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict fraud probability for a transaction",
    description="Analyze a single transaction and return fraud probability score",
    tags=["Fraud Detection"]
)
async def predict_fraud(
    request: Request,
    transaction: TransactionData,
    customer_id: str = Depends(verify_token)
):
    """
    Predict fraud probability for a single transaction.
    
    This endpoint processes a transaction in real-time and returns
    the fraud probability score along with risk assessment.
    
    Args:
        request: FastAPI request object
        transaction: Transaction data to analyze
        customer_id: Authenticated customer ID
        
    Returns:
        FraudPredictionResponse: Prediction results
        
    Raises:
        HTTPException: If prediction fails
    """
    start_time = datetime.utcnow()
    request_id = request.state.request_id
    
    logger.info(f"Processing fraud prediction request {request_id} for customer {customer_id}")
    
    try:
        # Generate transaction ID if not provided
        transaction_id = transaction.transaction_id or f"txn_{uuid.uuid4().hex[:12]}"
        
        # Get resources from app state
        xgb_model = request.app.state.resources.get('xgboost_model')
        feature_store = request.app.state.resources.get('feature_store')
        
        if not xgb_model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not available"
            )
        
        # Prepare features for prediction
        features = await prepare_features(transaction, feature_store, customer_id)
        
        # Make prediction
        fraud_probability = xgb_model.predict_proba(features)[1]
        
        # Determine risk level
        if fraud_probability < 0.3:
            risk_level = "low"
        elif fraud_probability < 0.6:
            risk_level = "medium"
        elif fraud_probability < 0.85:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Log prediction
        logger.info(
            f"Prediction completed - Transaction: {transaction_id}, "
            f"Probability: {fraud_probability:.4f}, Risk: {risk_level}"
        )
        
        return FraudPredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=float(fraud_probability),
            risk_level=risk_level,
            model_version="xgboost_v2.1.0",
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow(),
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post(
    "/explain",
    response_model=ExplainabilityResponse,
    summary="Get explanation for fraud prediction",
    description="Returns detailed explanation of why a transaction was flagged",
    tags=["Fraud Detection", "Explainability"]
)
async def explain_prediction(
    request: Request,
    transaction: TransactionData,
    customer_id: str = Depends(verify_token)
):
    """
    Get explanation for a fraud prediction.
    
    This endpoint provides interpretability results explaining which factors
    contributed most to the fraud score.
    
    Args:
        request: FastAPI request object
        transaction: Transaction data to explain
        customer_id: Authenticated customer ID
        
    Returns:
        ExplainabilityResponse: Explanation results
    """
    logger.info(f"Generating explanation for transaction")
    
    try:
        xgb_model = request.app.state.resources.get('xgboost_model')
        
        if not xgb_model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not available"
            )
        
        # Prepare features
        features = await prepare_features(transaction)
        
        # Get prediction
        fraud_probability = xgb_model.predict_proba(features)[1]
        
        # Get SHAP values for explanation
        shap_values = xgb_model.get_shap_values(features)
        
        # Identify top risk factors
        feature_names = features.columns.tolist()
        risk_factors = []
        
        for i, (name, value) in enumerate(zip(feature_names, shap_values)):
            if abs(value) > 0.1:  # Significant contribution
                risk_factors.append({
                    "factor": name,
                    "impact": float(abs(value)),
                    "direction": "increases" if value > 0 else "decreases",
                    "description": get_feature_description(name, value, features[name].values[0])
                })
        
        # Sort by impact
        risk_factors.sort(key=lambda x: x['impact'], reverse=True)
        
        return ExplainabilityResponse(
            transaction_id=transaction.transaction_id or f"txn_{uuid.uuid4().hex[:12]}",
            fraud_probability=float(fraud_probability),
            top_risk_factors=risk_factors[:5],
            feature_importance=dict(zip(feature_names, map(float, shap_values))),
            shap_values=dict(zip(feature_names, map(float, shap_values))),
            rule_triggers=get_triggered_rules(transaction)
        )
        
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )

@router.post(
    "/batch/predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch fraud prediction",
    description="Process multiple transactions for fraud detection",
    tags=["Fraud Detection", "Batch Processing"]
)
async def batch_predict(
    request: Request,
    batch_request: BatchPredictionRequest,
    customer_id: str = Depends(verify_token)
):
    """
    Process multiple transactions in a single request.
    
    For large volumes of transactions, this endpoint provides efficient
    batch processing with optional asynchronous handling.
    
    Args:
        request: FastAPI request object
        batch_request: Batch of transactions to process
        customer_id: Authenticated customer ID
    """
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    
    logger.info(f"Batch request {batch_id} received with {len(batch_request.transactions)} transactions")
    
    if batch_request.async_processing:
        # Handle asynchronously
        # In production, this would queue the job and return immediately
        return {
            "batch_id": batch_id,
            "status": "accepted",
            "message": "Batch accepted for async processing",
            "webhook_url": batch_request.webhook_url
        }
    else:
        # Process synchronously
        results = []
        total_risk = 0
        high_risk_count = 0
        
        for transaction in batch_request.transactions:
            try:
                # Process each transaction
                prediction = await predict_fraud(request, transaction, customer_id)
                results.append(prediction)
                total_risk += prediction.fraud_probability
                if prediction.fraud_probability > 0.7:
                    high_risk_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process transaction in batch {batch_id}: {str(e)}")
        
        return BatchPredictionResponse(
            batch_id=batch_id,
            total_transactions=len(batch_request.transactions),
            results=results,
            summary={
                "avg_risk_score": total_risk / len(results) if results else 0,
                "high_risk_count": high_risk_count,
                "successful_count": len(results),
                "failed_count": len(batch_request.transactions) - len(results),
                "processing_time_seconds": 0.5  # In production, calculate actual time
            }
        )

@router.get(
    "/model/info",
    summary="Get model information",
    description="Returns information about the currently deployed model",
    tags=["System", "Model Management"]
)
async def get_model_info(
    request: Request,
    customer_id: str = Depends(verify_token)
):
    """
    Get information about the deployed model.
    
    Returns model version, features used, and performance metrics.
    """
    return {
        "model_name": "FraudDetectionXGBoost",
        "version": "2.1.0",
        "deployed_at": "2024-01-01T00:00:00Z",
        "features": [
            "amount",
            "customer_avg_amount",
            "transaction_velocity",
            "device_risk_score",
            "location_risk"
        ],
        "performance_metrics": {
            "auc_roc": 0.92,
            "precision_at_0_5": 0.85,
            "recall_at_0_5": 0.78
        },
        "training_date": "2023-12-15",
        "training_samples": 1500000
    }

@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is healthy and ready to serve requests",
    tags=["System"]
)
async def health_check(request: Request):
    """
    Simple health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "up",
            "model": "up" if request.app.state.resources.get('xgboost_model') else "down"
        }
    }

# ============================================================================
# Helper Functions
# ============================================================================

async def prepare_features(transaction: TransactionData, feature_store=None, customer_id: str = None) -> Any:
    """
    Prepare features for model prediction.
    
    This function transforms raw transaction data into model-ready features
    by computing derived features and fetching historical data.
    
    Args:
        transaction: Raw transaction data
        feature_store: Feature store for retrieving historical features
        customer_id: Customer identifier
        
    Returns:
        DataFrame: Model-ready features
    """
    import pandas as pd
    
    # Convert to DataFrame
    features = pd.DataFrame([transaction.dict()])
    
    # Add derived features
    features['amount_log'] = np.log1p(features['amount'])
    features['hour_of_day'] = features['transaction_time'].dt.hour
    features['day_of_week'] = features['transaction_time'].dt.dayofweek
    features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
    
    # Add cyclical time features
    features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
    
    # Fetch customer features if available
    if feature_store and customer_id:
        customer_features = await feature_store.get_customer_features(customer_id)
        for key, value in customer_features.items():
            features[f'customer_{key}'] = value
    
    # Ensure all expected features are present
    expected_features = [
        'amount', 'amount_log', 'hour_sin', 'hour_cos', 'is_weekend',
        'customer_avg_amount', 'customer_std_amount', 'transaction_velocity_1h'
    ]
    
    for feature in expected_features:
        if feature not in features.columns:
            features[feature] = 0  # Default value
    
    return features[expected_features]

def get_feature_description(feature_name: str, shap_value: float, feature_value: float) -> str:
    """
    Generate human-readable description for a feature's contribution.
    
    Args:
        feature_name: Name of the feature
        shap_value: SHAP value for this feature
        feature_value: Actual feature value
        
    Returns:
        str: Human-readable description
    """
    descriptions = {
        'amount': lambda v, sv: f"Transaction amount of ${v:.2f} {'increases' if sv > 0 else 'decreases'} fraud probability",
        'amount_log': lambda v, sv: f"Log-transformed amount of {v:.2f} suggests {'higher' if sv > 0 else 'lower'} risk",
        'transaction_velocity_1h': lambda v, sv: f"{v} transactions in last hour - {'high' if sv > 0 else 'normal'} velocity",
        'customer_avg_amount': lambda v, sv: f"Amount is {v:.1f}x customer's average - {'unusual' if sv > 0 else 'typical'}",
        'device_risk_score': lambda v, sv: f"Device risk score {v:.2f} - {'suspicious' if sv > 0 else 'trusted'} device",
        'is_weekend': lambda v, sv: f"Transaction on {'weekend' if v else 'weekday'} - {'higher' if sv > 0 else 'typical'} risk pattern"
    }
    
    if feature_name in descriptions:
        return descriptions[feature_name](feature_value, shap_value)
    else:
        return f"Feature {feature_name} has {'positive' if shap_value > 0 else 'negative'} impact of {abs(shap_value):.3f}"

def get_triggered_rules(transaction: TransactionData) -> List[str]:
    """
    Identify business rules triggered by the transaction.
    
    Args:
        transaction: Transaction data
        
    Returns:
        List[str]: List of triggered rule names
    """
    triggered = []
    
    # Rule 1: High amount threshold
    if transaction.amount > 10000:
        triggered.append("HIGH_AMOUNT_THRESHOLD")
    
    # Rule 2: Weekend high amount
    if transaction.transaction_time.weekday() >= 5 and transaction.amount > 5000:
        triggered.append("WEEKEND_HIGH_AMOUNT")
    
    # Rule 3: International transaction (simplified)
    if transaction.country and transaction.country != "US":
        triggered.append("INTERNATIONAL_TRANSACTION")
    
    # Rule 4: New device (would need historical data)
    # triggered.append("NEW_DEVICE_DETECTED")
    
    return triggered