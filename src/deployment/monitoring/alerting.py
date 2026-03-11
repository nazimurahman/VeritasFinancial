# =============================================================================
# VERITASFINANCIAL - FRAUD DETECTION SYSTEM
# Module: deployment/monitoring/alerting.py
# Description: Real-time monitoring and alerting system for fraud detection
# Author: Data Science Team
# Version: 2.0.0
# Last Updated: 2024-01-15
# =============================================================================

"""
Alerting System for Banking Fraud Detection
============================================
This module provides comprehensive monitoring and alerting capabilities:
- Real-time metric monitoring
- Multi-channel alerts (Email, Slack, SMS, Teams)
- Threshold-based and anomaly-based alerting
- Alert aggregation and deduplication
- Escalation policies
- Alert history and analytics
"""

import asyncio
import logging
import smtplib
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from pathlib import Path

# Third-party imports with error handling
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests library not available. HTTP alerts disabled.")

try:
    import slack_sdk
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logging.warning("Slack SDK not available. Slack alerts disabled.")

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logging.warning("Boto3 not available. AWS services disabled.")

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Metrics export disabled.")

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alerting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Alert severity levels - define the importance/urgency of alerts
class AlertSeverity(Enum):
    """
    Alert severity levels based on impact and urgency.
    Used to prioritize alert handling and determine notification channels.
    """
    CRITICAL = 5    # System down, immediate action required, page on-call
    HIGH = 4        # Major functionality affected, respond within 15 min
    MEDIUM = 3      # Partial impact, respond within 1 hour
    LOW = 2         # Minor issues, respond within 24 hours
    INFO = 1        # Informational only, no action required
    DEBUG = 0       # Debugging purposes only

# Alert types - categorize alerts for better routing and analysis
class AlertType(Enum):
    """
    Categories of alerts for proper routing and analysis.
    Each type has different handling requirements and owners.
    """
    MODEL_PERFORMANCE = "model_performance"      # Model accuracy, drift, etc.
    DATA_QUALITY = "data_quality"                 # Data issues, missing values
    SYSTEM_HEALTH = "system_health"               # Infrastructure issues
    BUSINESS_METRIC = "business_metric"          # Business KPIs
    FRAUD_RATE = "fraud_rate"                     # Fraud rate anomalies
    LATENCY = "latency"                           # Response time issues
    THROUGHPUT = "throughput"                     # Transaction volume issues
    ERROR_RATE = "error_rate"                      # Error rate spikes
    SECURITY = "security"                          # Security incidents
    COMPLIANCE = "compliance"                       # Compliance violations

# Alert channels - where to send notifications
class AlertChannel(Enum):
    """
    Available notification channels for alerts.
    Multiple channels can be used based on severity and type.
    """
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    LOG = "log"

# =============================================================================
# DATA CLASSES FOR ALERT MANAGEMENT
# =============================================================================

@dataclass
class Alert:
    """
    Core alert data structure representing a single alert instance.
    
    Attributes:
        alert_id: Unique identifier for the alert
        name: Human-readable alert name
        description: Detailed description of the alert condition
        severity: AlertSeverity enum value
        alert_type: AlertType enum value
        source: Component/service that generated the alert
        timestamp: When the alert was generated
        value: Current metric value (if applicable)
        threshold: Threshold that was breached (if applicable)
        details: Additional context as key-value pairs
        tags: Categorization tags for filtering
        acknowledged: Whether alert has been acknowledged
        resolved: Whether alert condition has been resolved
        acknowledgement_time: When alert was acknowledged
        resolution_time: When alert was resolved
        escalation_level: Current escalation level (0 = none)
    """
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    alert_type: AlertType
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    value: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    acknowledgement_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None
    escalation_level: int = 0
    
    def __post_init__(self):
        """Validate and initialize alert after creation."""
        if not self.alert_id:
            # Generate unique ID based on content for deduplication
            content = f"{self.name}{self.source}{self.timestamp}"
            self.alert_id = hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        result = asdict(self)
        # Convert enums to strings for JSON serialization
        result['severity'] = self.severity.value
        result['alert_type'] = self.alert_type.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.acknowledgement_time:
            result['acknowledgement_time'] = self.acknowledgement_time.isoformat()
        if self.resolution_time:
            result['resolution_time'] = self.resolution_time.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary."""
        # Convert string values back to enums
        data['severity'] = AlertSeverity(data['severity'])
        data['alert_type'] = AlertType(data['alert_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('acknowledgement_time'):
            data['acknowledgement_time'] = datetime.fromisoformat(data['acknowledgement_time'])
        if data.get('resolution_time'):
            data['resolution_time'] = datetime.fromisoformat(data['resolution_time'])
        return cls(**data)


@dataclass
class AlertRule:
    """
    Rule definition for generating alerts based on conditions.
    
    Attributes:
        rule_id: Unique identifier for the rule
        name: Human-readable rule name
        description: Rule description
        metric_name: Name of metric to monitor
        condition: Comparison condition (gt, lt, eq, etc.)
        threshold: Threshold value
        severity: Alert severity when rule triggers
        alert_type: Type of alert
        window_seconds: Time window for evaluation
        cooldown_seconds: Minimum time between alerts
        channels: List of channels to notify
        enabled: Whether rule is active
        tags: Categorization tags
    """
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'gte', 'lte', 'ne', 'within_range', 'outside_range'
    threshold: Union[float, tuple]
    severity: AlertSeverity
    alert_type: AlertType
    window_seconds: int = 60
    cooldown_seconds: int = 300
    channels: List[AlertChannel] = field(default_factory=list)
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    
    def evaluate(self, value: float) -> bool:
        """
        Evaluate if the condition is met.
        
        Args:
            value: Current metric value
            
        Returns:
            True if condition met, False otherwise
        """
        if self.condition == 'gt':
            return value > self.threshold
        elif self.condition == 'lt':
            return value < self.threshold
        elif self.condition == 'eq':
            return value == self.threshold
        elif self.condition == 'gte':
            return value >= self.threshold
        elif self.condition == 'lte':
            return value <= self.threshold
        elif self.condition == 'ne':
            return value != self.threshold
        elif self.condition == 'within_range':
            # threshold is tuple (min, max)
            min_val, max_val = self.threshold
            return min_val <= value <= max_val
        elif self.condition == 'outside_range':
            min_val, max_val = self.threshold
            return value < min_val or value > max_val
        else:
            logger.error(f"Unknown condition: {self.condition}")
            return False


# =============================================================================
# NOTIFICATION CHANNEL HANDLERS
# =============================================================================

class NotificationChannel:
    """
    Base class for all notification channels.
    Defines the interface that all channel implementations must follow.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize notification channel.
        
        Args:
            name: Channel name
            config: Channel configuration dictionary
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.rate_limit_seconds = config.get('rate_limit_seconds', 0)
        self.last_sent = defaultdict(lambda: datetime.min)
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def send(self, alert: Alert, rule: Optional[AlertRule] = None) -> bool:
        """
        Send alert through this channel.
        
        Args:
            alert: Alert to send
            rule: Rule that triggered the alert
            
        Returns:
            True if sent successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement send()")
    
    def _check_rate_limit(self, alert: Alert) -> bool:
        """
        Check if we've exceeded rate limit for this alert type.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if rate limit not exceeded, False otherwise
        """
        if self.rate_limit_seconds <= 0:
            return True
        
        key = f"{alert.alert_type.value}_{alert.severity.value}"
        time_since_last = (datetime.now() - self.last_sent[key]).total_seconds()
        
        if time_since_last < self.rate_limit_seconds:
            self.logger.warning(f"Rate limit exceeded for {key}")
            return False
        
        self.last_sent[key] = datetime.now()
        return True


class SlackChannel(NotificationChannel):
    """
    Slack notification channel.
    Sends alerts to Slack channels with rich formatting.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize Slack channel.
        
        Args:
            name: Channel name
            config: Configuration with webhook_url, channel, etc.
        """
        super().__init__(name, config)
        
        if not SLACK_AVAILABLE:
            self.enabled = False
            self.logger.error("Slack SDK not available")
            return
        
        self.webhook_url = config.get('webhook_url')
        self.token = config.get('token')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'FraudGuard Alert')
        self.icon_emoji = config.get('icon_emoji', ':warning:')
        
        # Initialize Slack client if token provided
        if self.token:
            self.client = slack_sdk.WebClient(token=self.token)
        else:
            self.client = None
    
    async def send(self, alert: Alert, rule: Optional[AlertRule] = None) -> bool:
        """
        Send alert to Slack.
        
        Args:
            alert: Alert to send
            rule: Rule that triggered the alert
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            self.logger.debug(f"Channel {self.name} disabled")
            return False
        
        if not self._check_rate_limit(alert):
            return False
        
        try:
            # Format message based on severity
            message = self._format_alert(alert, rule)
            
            # Send via webhook or client
            if self.webhook_url:
                response = requests.post(
                    self.webhook_url,
                    json=message,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
            elif self.client:
                response = self.client.chat_postMessage(
                    channel=self.channel,
                    **message
                )
            else:
                self.logger.error("No Slack configuration provided")
                return False
            
            self.logger.info(f"Alert {alert.alert_id} sent to Slack")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send alert to Slack: {e}")
            return False
    
    def _format_alert(self, alert: Alert, rule: Optional[AlertRule]) -> Dict[str, Any]:
        """
        Format alert for Slack message.
        
        Args:
            alert: Alert to format
            rule: Rule that triggered the alert
            
        Returns:
            Slack message payload
        """
        # Color based on severity
        colors = {
            AlertSeverity.CRITICAL: "#FF0000",  # Red
            AlertSeverity.HIGH: "#FFA500",       # Orange
            AlertSeverity.MEDIUM: "#FFFF00",     # Yellow
            AlertSeverity.LOW: "#0000FF",        # Blue
            AlertSeverity.INFO: "#808080",       # Gray
            AlertSeverity.DEBUG: "#00FF00"       # Green
        }
        
        # Create attachment
        attachment = {
            "color": colors.get(alert.severity, "#000000"),
            "title": f"[{alert.severity.name}] {alert.name}",
            "text": alert.description,
            "fields": [
                {
                    "title": "Alert ID",
                    "value": alert.alert_id,
                    "short": True
                },
                {
                    "title": "Type",
                    "value": alert.alert_type.value,
                    "short": True
                },
                {
                    "title": "Source",
                    "value": alert.source,
                    "short": True
                },
                {
                    "title": "Time",
                    "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "short": True
                }
            ],
            "footer": "VeritasFinancial Fraud Detection System",
            "ts": int(alert.timestamp.timestamp())
        }
        
        # Add value if available
        if alert.value is not None:
            attachment["fields"].append({
                "title": "Current Value",
                "value": str(alert.value),
                "short": True
            })
        
        # Add threshold if available
        if alert.threshold is not None:
            attachment["fields"].append({
                "title": "Threshold",
                "value": str(alert.threshold),
                "short": True
            })
        
        # Add rule info if available
        if rule:
            attachment["fields"].append({
                "title": "Rule",
                "value": rule.name,
                "short": True
            })
        
        # Add any additional details
        if alert.details:
            for key, value in list(alert.details.items())[:5]:  # Limit to 5 fields
                attachment["fields"].append({
                    "title": key.replace('_', ' ').title(),
                    "value": str(value),
                    "short": True
                })
        
        return {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [attachment]
        }


class EmailChannel(NotificationChannel):
    """
    Email notification channel.
    Sends alerts via SMTP with HTML formatting.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize Email channel.
        
        Args:
            name: Channel name
            config: Configuration with SMTP settings, recipients, etc.
        """
        super().__init__(name, config)
        
        self.smtp_host = config.get('smtp_host', 'localhost')
        self.smtp_port = config.get('smtp_port', 25)
        self.smtp_user = config.get('smtp_user')
        self.smtp_password = config.get('smtp_password')
        self.use_tls = config.get('use_tls', True)
        self.from_addr = config.get('from_addr', 'alerts@veritasfinancial.com')
        self.to_addrs = config.get('to_addrs', [])
        self.cc_addrs = config.get('cc_addrs', [])
        self.bcc_addrs = config.get('bcc_addrs', [])
    
    async def send(self, alert: Alert, rule: Optional[AlertRule] = None) -> bool:
        """
        Send alert via email.
        
        Args:
            alert: Alert to send
            rule: Rule that triggered the alert
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        if not self._check_rate_limit(alert):
            return False
        
        if not self.to_addrs:
            self.logger.error("No recipients configured")
            return False
        
        try:
            # Create email message
            msg = self._create_email(alert, rule)
            
            # Send via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            self.logger.info(f"Alert {alert.alert_id} sent via email")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_email(self, alert: Alert, rule: Optional[AlertRule]):
        """
        Create email message from alert.
        
        Args:
            alert: Alert to format
            rule: Rule that triggered the alert
            
        Returns:
            EmailMessage object
        """
        from email.message import EmailMessage
        
        # Create message
        msg = EmailMessage()
        msg.set_content(self._format_text(alert, rule))
        msg.add_alternative(self._format_html(alert, rule), subtype='html')
        
        # Set headers
        msg['Subject'] = f"[{alert.severity.name}] {alert.name}"
        msg['From'] = self.from_addr
        msg['To'] = ', '.join(self.to_addrs)
        if self.cc_addrs:
            msg['Cc'] = ', '.join(self.cc_addrs)
        
        return msg
    
    def _format_text(self, alert: Alert, rule: Optional[AlertRule]) -> str:
        """Format plain text version of email."""
        lines = []
        lines.append(f"ALERT: {alert.name}")
        lines.append("=" * 50)
        lines.append(f"Severity: {alert.severity.name}")
        lines.append(f"Type: {alert.alert_type.value}")
        lines.append(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Source: {alert.source}")
        lines.append("")
        lines.append("Description:")
        lines.append(alert.description)
        lines.append("")
        
        if alert.value is not None:
            lines.append(f"Current Value: {alert.value}")
        if alert.threshold is not None:
            lines.append(f"Threshold: {alert.threshold}")
        if rule:
            lines.append(f"Rule: {rule.name}")
        
        if alert.details:
            lines.append("")
            lines.append("Additional Details:")
            for key, value in alert.details.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def _format_html(self, alert: Alert, rule: Optional[AlertRule]) -> str:
        """Format HTML version of email."""
        # Color mapping for severity
        colors = {
            AlertSeverity.CRITICAL: '#dc3545',  # Red
            AlertSeverity.HIGH: '#fd7e14',      # Orange
            AlertSeverity.MEDIUM: '#ffc107',    # Yellow
            AlertSeverity.LOW: '#0d6efd',       # Blue
            AlertSeverity.INFO: '#6c757d',      # Gray
            AlertSeverity.DEBUG: '#198754'      # Green
        }
        
        color = colors.get(alert.severity, '#000000')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: {color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .field {{ margin: 10px 0; }}
                .label {{ font-weight: bold; color: #666; }}
                .value {{ margin-left: 10px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Alert: {alert.name}</h1>
                <p>Severity: {alert.severity.name}</p>
            </div>
            <div class="content">
                <div class="field">
                    <span class="label">Alert ID:</span>
                    <span class="value">{alert.alert_id}</span>
                </div>
                <div class="field">
                    <span class="label">Type:</span>
                    <span class="value">{alert.alert_type.value}</span>
                </div>
                <div class="field">
                    <span class="label">Time:</span>
                    <span class="value">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span>
                </div>
                <div class="field">
                    <span class="label">Source:</span>
                    <span class="value">{alert.source}</span>
                </div>
                
                <h2>Description</h2>
                <p>{alert.description}</p>
                
                <h2>Details</h2>
                <table>
        """
        
        if alert.value is not None:
            html += f"""
                    <tr>
                        <th>Current Value</th>
                        <td>{alert.value}</td>
                    </tr>
            """
        
        if alert.threshold is not None:
            html += f"""
                    <tr>
                        <th>Threshold</th>
                        <td>{alert.threshold}</td>
                    </tr>
            """
        
        if rule:
            html += f"""
                    <tr>
                        <th>Rule</th>
                        <td>{rule.name}</td>
                    </tr>
                    <tr>
                        <th>Rule Description</th>
                        <td>{rule.description}</td>
                    </tr>
            """
        
        for key, value in alert.details.items():
            html += f"""
                    <tr>
                        <th>{key.replace('_', ' ').title()}</th>
                        <td>{value}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html


class WebhookChannel(NotificationChannel):
    """
    Generic webhook notification channel.
    Sends alerts to any HTTP endpoint.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize Webhook channel.
        
        Args:
            name: Channel name
            config: Configuration with url, headers, etc.
        """
        super().__init__(name, config)
        
        if not REQUESTS_AVAILABLE:
            self.enabled = False
            self.logger.error("Requests library not available")
            return
        
        self.url = config.get('url')
        self.method = config.get('method', 'POST')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 10)
        self.retry_count = config.get('retry_count', 3)
        
        # Configure retry strategy
        if self.retry_count > 0:
            retry_strategy = Retry(
                total=self.retry_count,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session = requests.Session()
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        else:
            self.session = requests.Session()
    
    async def send(self, alert: Alert, rule: Optional[AlertRule] = None) -> bool:
        """
        Send alert to webhook.
        
        Args:
            alert: Alert to send
            rule: Rule that triggered the alert
            
        Returns:
            True if sent successfully
        """
        if not self.enabled or not self.url:
            return False
        
        if not self._check_rate_limit(alert):
            return False
        
        try:
            # Prepare payload
            payload = self._prepare_payload(alert, rule)
            
            # Send request
            response = self.session.request(
                method=self.method,
                url=self.url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            self.logger.info(f"Alert {alert.alert_id} sent to webhook")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def _prepare_payload(self, alert: Alert, rule: Optional[AlertRule]) -> Dict[str, Any]:
        """Prepare payload for webhook."""
        payload = {
            'alert_id': alert.alert_id,
            'name': alert.name,
            'description': alert.description,
            'severity': alert.severity.value,
            'severity_name': alert.severity.name,
            'type': alert.alert_type.value,
            'source': alert.source,
            'timestamp': alert.timestamp.isoformat(),
            'value': alert.value,
            'threshold': alert.threshold,
            'details': alert.details,
            'tags': alert.tags
        }
        
        if rule:
            payload['rule'] = {
                'rule_id': rule.rule_id,
                'name': rule.name,
                'description': rule.description,
                'metric_name': rule.metric_name,
                'condition': rule.condition,
                'threshold': rule.threshold
            }
        
        return payload


class ConsoleChannel(NotificationChannel):
    """
    Console notification channel.
    Prints alerts to console for debugging.
    """
    
    async def send(self, alert: Alert, rule: Optional[AlertRule] = None) -> bool:
        """
        Print alert to console.
        
        Args:
            alert: Alert to send
            rule: Rule that triggered the alert
            
        Returns:
            True always
        """
        if not self.enabled:
            return False
        
        # Color codes for console output
        colors = {
            AlertSeverity.CRITICAL: '\033[91m',  # Red
            AlertSeverity.HIGH: '\033[93m',      # Yellow
            AlertSeverity.MEDIUM: '\033[94m',    # Blue
            AlertSeverity.LOW: '\033[92m',       # Green
            AlertSeverity.INFO: '\033[90m',      # Gray
            AlertSeverity.DEBUG: '\033[95m'       # Purple
        }
        reset = '\033[0m'
        
        color = colors.get(alert.severity, '')
        
        print(f"\n{color}{'='*60}{reset}")
        print(f"{color}[{alert.severity.name}] {alert.name}{reset}")
        print(f"{color}{'='*60}{reset}")
        print(f"ID: {alert.alert_id}")
        print(f"Type: {alert.alert_type.value}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Source: {alert.source}")
        print(f"\nDescription: {alert.description}")
        
        if alert.value is not None:
            print(f"Value: {alert.value}")
        if alert.threshold is not None:
            print(f"Threshold: {alert.threshold}")
        if rule:
            print(f"Rule: {rule.name}")
        
        if alert.details:
            print("\nDetails:")
            for key, value in alert.details.items():
                print(f"  {key}: {value}")
        
        print(f"{color}{'='*60}{reset}\n")
        
        return True


# =============================================================================
# ALERT MANAGER - CORE ALERTING SYSTEM
# =============================================================================

class AlertManager:
    """
    Central alert management system.
    
    Responsibilities:
    - Rule evaluation and alert generation
    - Alert deduplication and aggregation
    - Multi-channel notification routing
    - Alert history and analytics
    - Escalation policies
    - Integration with monitoring systems
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AlertManager.
        
        Args:
            config_path: Path to configuration file (JSON/YAML)
        """
        self.logger = logging.getLogger(f"{__name__}.AlertManager")
        
        # Core components
        self.rules: Dict[str, AlertRule] = {}           # All active rules
        self.channels: Dict[str, NotificationChannel] = {}  # Notification channels
        self.active_alerts: Dict[str, Alert] = {}       # Currently active alerts
        self.alert_history: deque = deque(maxlen=10000)  # Recent alerts (limit memory)
        self.rule_last_trigger: Dict[str, datetime] = {}  # Last trigger time per rule
        self.alert_counts: Dict[str, int] = defaultdict(int)  # Alert frequency counts
        
        # Configuration
        self.config = self._load_config(config_path) if config_path else {}
        self.escalation_policies = self.config.get('escalation_policies', [])
        self.aggregation_window = self.config.get('aggregation_window_seconds', 300)
        self.max_alerts_per_rule = self.config.get('max_alerts_per_rule_per_hour', 10)
        
        # Metrics for Prometheus
        if PROMETHEUS_AVAILABLE:
            self.metrics = {
                'alerts_total': Counter('alerts_total', 'Total alerts generated', ['severity', 'type']),
                'active_alerts': Gauge('active_alerts', 'Currently active alerts', ['severity']),
                'alert_duration': Histogram('alert_duration_seconds', 'Alert resolution time'),
                'rules_evaluated': Counter('rules_evaluated_total', 'Total rule evaluations')
            }
        
        # Initialize default channels
        self._init_default_channels()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("AlertManager initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        config = {}
        path = Path(config_path)
        
        try:
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    config = json.load(f)
            elif path.suffix in ['.yaml', '.yml']:
                import yaml
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                self.logger.error(f"Unsupported config file format: {path.suffix}")
            
            self.logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
        
        return config
    
    def _init_default_channels(self):
        """Initialize default notification channels."""
        channel_configs = self.config.get('channels', {})
        
        # Slack channel
        if 'slack' in channel_configs:
            self.add_channel('slack', SlackChannel('slack', channel_configs['slack']))
        
        # Email channel
        if 'email' in channel_configs:
            self.add_channel('email', EmailChannel('email', channel_configs['email']))
        
        # Webhook channel
        if 'webhook' in channel_configs:
            self.add_channel('webhook', WebhookChannel('webhook', channel_configs['webhook']))
        
        # Console channel (always add for debugging)
        self.add_channel('console', ConsoleChannel('console', {'enabled': True}))
    
    def _start_background_tasks(self):
        """Start background tasks for alert management."""
        # This would typically use asyncio or threading
        # For simplicity, we'll note that these tasks should be started
        self.logger.info("Background tasks configured")
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        Add a new alert rule.
        
        Args:
            rule: AlertRule to add
        """
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Added rule: {rule.name} ({rule.rule_id})")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            True if removed, False if not found
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Removed rule: {rule_id}")
            return True
        return False
    
    def add_channel(self, name: str, channel: NotificationChannel) -> None:
        """
        Add a notification channel.
        
        Args:
            name: Channel name
            channel: NotificationChannel instance
        """
        self.channels[name] = channel
        self.logger.info(f"Added channel: {name}")
    
    def remove_channel(self, name: str) -> bool:
        """
        Remove a notification channel.
        
        Args:
            name: Channel name
            
        Returns:
            True if removed, False if not found
        """
        if name in self.channels:
            del self.channels[name]
            self.logger.info(f"Removed channel: {name}")
            return True
        return False
    
    def evaluate_metrics(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Evaluate metrics against all rules.
        
        Args:
            metrics: Dictionary of metric_name -> value
            
        Returns:
            List of generated alerts
        """
        generated_alerts = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Update metrics
            if PROMETHEUS_AVAILABLE:
                self.metrics['rules_evaluated'].inc()
            
            # Check if metric exists
            if rule.metric_name not in metrics:
                continue
            
            value = metrics[rule.metric_name]
            
            # Check if rule should trigger
            if self._should_trigger_rule(rule, value):
                alert = self._create_alert(rule, value, metrics)
                generated_alerts.append(alert)
                self._process_alert(alert, rule)
        
        return generated_alerts
    
    def _should_trigger_rule(self, rule: AlertRule, value: float) -> bool:
        """
        Check if rule should trigger based on value and cooldown.
        
        Args:
            rule: Rule to check
            value: Current metric value
            
        Returns:
            True if should trigger, False otherwise
        """
        # Check if rule condition is met
        if not rule.evaluate(value):
            return False
        
        # Check cooldown
        last_trigger = self.rule_last_trigger.get(rule.rule_id)
        if last_trigger:
            time_since = (datetime.now() - last_trigger).total_seconds()
            if time_since < rule.cooldown_seconds:
                return False
        
        # Check rate limit
        hour_key = f"{rule.rule_id}_{datetime.now().strftime('%Y%m%d%H')}"
        if self.alert_counts[hour_key] >= self.max_alerts_per_rule:
            self.logger.warning(f"Rate limit exceeded for rule {rule.rule_id}")
            return False
        
        return True
    
    def _create_alert(self, rule: AlertRule, value: float, metrics: Dict[str, float]) -> Alert:
        """
        Create alert from rule and value.
        
        Args:
            rule: Rule that triggered
            value: Current metric value
            metrics: All metrics for context
            
        Returns:
            Created Alert
        """
        # Prepare details from metrics
        details = {}
        for name, val in metrics.items():
            if name != rule.metric_name:
                details[name] = val
        
        # Create alert
        alert = Alert(
            name=f"{rule.name} - Alert",
            description=f"Rule '{rule.name}' triggered with value {value:.4f}",
            severity=rule.severity,
            alert_type=rule.alert_type,
            source="AlertManager",
            value=value,
            threshold=rule.threshold if isinstance(rule.threshold, float) else None,
            details=details,
            tags=rule.tags
        )
        
        return alert
    
    def _process_alert(self, alert: Alert, rule: AlertRule) -> None:
        """
        Process a generated alert.
        
        Args:
            alert: Alert to process
            rule: Rule that generated it
        """
        # Update tracking
        self.rule_last_trigger[rule.rule_id] = datetime.now()
        
        hour_key = f"{rule.rule_id}_{datetime.now().strftime('%Y%m%d%H')}"
        self.alert_counts[hour_key] += 1
        
        # Add to active alerts if not resolved
        if not alert.resolved:
            self.active_alerts[alert.alert_id] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Update metrics
        if PROMETHEUS_AVAILABLE:
            self.metrics['alerts_total'].labels(
                severity=alert.severity.name,
                type=alert.alert_type.value
            ).inc()
            self.metrics['active_alerts'].labels(severity=alert.severity.name).inc()
        
        # Send notifications
        asyncio.create_task(self._notify_channels(alert, rule))
        
        self.logger.info(f"Processed alert {alert.alert_id} from rule {rule.rule_id}")
    
    async def _notify_channels(self, alert: Alert, rule: AlertRule) -> None:
        """
        Send alert to configured channels.
        
        Args:
            alert: Alert to send
            rule: Rule that generated it
        """
        # Determine channels based on rule or severity
        channels = rule.channels if rule.channels else self._get_channels_for_severity(alert.severity)
        
        # Send to each channel
        tasks = []
        for channel_name in channels:
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                tasks.append(channel.send(alert, rule))
        
        # Wait for all notifications
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Channel notification failed: {result}")
    
    def _get_channels_for_severity(self, severity: AlertSeverity) -> List[str]:
        """
        Get default channels for severity level.
        
        Args:
            severity: Alert severity
            
        Returns:
            List of channel names
        """
        severity_channels = {
            AlertSeverity.CRITICAL: ['slack', 'email', 'console'],
            AlertSeverity.HIGH: ['slack', 'email', 'console'],
            AlertSeverity.MEDIUM: ['slack', 'console'],
            AlertSeverity.LOW: ['slack', 'console'],
            AlertSeverity.INFO: ['console'],
            AlertSeverity.DEBUG: ['console']
        }
        
        return severity_channels.get(severity, ['console'])
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            user: User acknowledging the alert
            
        Returns:
            True if acknowledged, False if not found
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledgement_time = datetime.now()
            alert.details['acknowledged_by'] = user
            
            self.logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolution: str = "") -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: Alert ID to resolve
            resolution: Resolution notes
            
        Returns:
            True if resolved, False if not found
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            if resolution:
                alert.details['resolution'] = resolution
            
            # Calculate duration
            duration = (alert.resolution_time - alert.timestamp).total_seconds()
            if PROMETHEUS_AVAILABLE:
                self.metrics['alert_duration'].observe(duration)
                self.metrics['active_alerts'].labels(severity=alert.severity.name).dec()
            
            # Remove from active
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert {alert_id} resolved after {duration:.2f} seconds")
            
            # Send resolution notification
            asyncio.create_task(self._notify_resolution(alert))
            
            return True
        
        return False
    
    async def _notify_resolution(self, alert: Alert) -> None:
        """Notify that an alert has been resolved."""
        resolution_alert = Alert(
            name=f"{alert.name} - RESOLVED",
            description=f"Alert resolved: {alert.description}",
            severity=AlertSeverity.INFO,
            alert_type=alert.alert_type,
            source=alert.source,
            details={**alert.details, 'original_alert_id': alert.alert_id},
            tags=alert.tags + ['resolved']
        )
        
        await self._notify_channels(resolution_alert, None)
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """
        Get currently active alerts.
        
        Args:
            severity: Filter by severity
            
        Returns:
            List of active alerts
        """
        if severity:
            return [a for a in self.active_alerts.values() if a.severity == severity]
        return list(self.active_alerts.values())
    
    def get_alert_history(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         alert_type: Optional[AlertType] = None,
                         severity: Optional[AlertSeverity] = None,
                         limit: int = 100) -> List[Alert]:
        """
        Get alert history with filters.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            alert_type: Filter by alert type
            severity: Filter by severity
            limit: Maximum number of alerts to return
            
        Returns:
            List of filtered alerts
        """
        alerts = list(self.alert_history)
        
        # Apply filters
        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Sort by timestamp (newest first) and limit
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]
    
    def get_alert_stats(self, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            
        Returns:
            Dictionary with alert statistics
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        alerts = self.get_alert_history(start_time, end_time)
        
        stats = {
            'total_alerts': len(alerts),
            'by_severity': defaultdict(int),
            'by_type': defaultdict(int),
            'by_source': defaultdict(int),
            'avg_resolution_time': 0,
            'active_count': len(self.active_alerts),
            'most_frequent_rules': [],
            'time_series': defaultdict(int)
        }
        
        resolution_times = []
        
        for alert in alerts:
            stats['by_severity'][alert.severity.name] += 1
            stats['by_type'][alert.alert_type.value] += 1
            stats['by_source'][alert.source] += 1
            
            # Time series by hour
            hour_key = alert.timestamp.strftime('%Y-%m-%d %H:00')
            stats['time_series'][hour_key] += 1
            
            # Resolution time
            if alert.resolution_time:
                duration = (alert.resolution_time - alert.timestamp).total_seconds()
                resolution_times.append(duration)
        
        if resolution_times:
            stats['avg_resolution_time'] = sum(resolution_times) / len(resolution_times)
            stats['min_resolution_time'] = min(resolution_times)
            stats['max_resolution_time'] = max(resolution_times)
        
        return dict(stats)
    
    def export_metrics(self) -> Optional[str]:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus metrics string or None
        """
        if PROMETHEUS_AVAILABLE:
            return generate_latest()
        return None


# =============================================================================
# FACTORY FUNCTION FOR EASY CREATION
# =============================================================================

def create_alert_manager(config_path: Optional[str] = None) -> AlertManager:
    """
    Factory function to create and configure AlertManager.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured AlertManager instance
    """
    manager = AlertManager(config_path)
    
    # Add default rules if no config provided
    if not config_path:
        # Fraud rate spike rule
        manager.add_rule(AlertRule(
            rule_id="fraud_rate_spike",
            name="Fraud Rate Spike",
            description="Alert when fraud rate exceeds threshold",
            metric_name="fraud_rate",
            condition="gt",
            threshold=0.05,  # 5% fraud rate
            severity=AlertSeverity.HIGH,
            alert_type=AlertType.FRAUD_RATE,
            window_seconds=300,
            cooldown_seconds=600,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            tags=['fraud', 'real-time']
        ))
        
        # Model performance degradation
        manager.add_rule(AlertRule(
            rule_id="model_auc_drop",
            name="Model AUC Drop",
            description="Alert when model AUC drops below threshold",
            metric_name="model_auc",
            condition="lt",
            threshold=0.85,
            severity=AlertSeverity.HIGH,
            alert_type=AlertType.MODEL_PERFORMANCE,
            window_seconds=3600,
            cooldown_seconds=1800,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            tags=['model', 'performance']
        ))
        
        # High latency
        manager.add_rule(AlertRule(
            rule_id="high_latency",
            name="High Inference Latency",
            description="Alert when inference latency exceeds threshold",
            metric_name="inference_latency_ms",
            condition="gt",
            threshold=500,  # 500ms
            severity=AlertSeverity.MEDIUM,
            alert_type=AlertType.LATENCY,
            window_seconds=60,
            cooldown_seconds=300,
            channels=[AlertChannel.SLACK],
            tags=['performance', 'api']
        ))
        
        # Data quality issue
        manager.add_rule(AlertRule(
            rule_id="missing_data",
            name="Missing Data Detected",
            description="Alert when missing data rate exceeds threshold",
            metric_name="missing_data_rate",
            condition="gt",
            threshold=0.1,  # 10% missing
            severity=AlertSeverity.MEDIUM,
            alert_type=AlertType.DATA_QUALITY,
            window_seconds=3600,
            cooldown_seconds=3600,
            channels=[AlertChannel.SLACK],
            tags=['data-quality']
        ))
    
    return manager


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the alerting system.
    """
    import asyncio
    
    async def example():
        # Create alert manager
        manager = create_alert_manager()
        
        # Simulate metrics
        metrics = {
            'fraud_rate': 0.07,  # 7% (above threshold)
            'model_auc': 0.88,
            'inference_latency_ms': 450,
            'missing_data_rate': 0.05,
            'transaction_volume': 1500
        }
        
        # Evaluate metrics
        alerts = manager.evaluate_metrics(metrics)
        
        print(f"Generated {len(alerts)} alerts")
        
        # Wait for notifications
        await asyncio.sleep(2)
        
        # Check active alerts
        active = manager.get_active_alerts()
        print(f"Active alerts: {len(active)}")
        
        # Get statistics
        stats = manager.get_alert_stats()
        print(f"Alert stats: {json.dumps(stats, indent=2, default=str)}")
        
        # Resolve alerts if needed
        for alert in active:
            if alert.alert_type == AlertType.FRAUD_RATE:
                manager.resolve_alert(alert.alert_id, "Investigated - pattern normal")
        
        return manager
    
    # Run example
    manager = asyncio.run(example())