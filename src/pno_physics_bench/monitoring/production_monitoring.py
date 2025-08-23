# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials

"""
Production-Grade Monitoring and Observability System
Generation 2 Robustness Enhancement
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import weakref

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics for categorization."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    TIMER = auto()


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Metric:
    """Represents a metric measurement."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Represents a system alert."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    severity: AlertSeverity = AlertSeverity.MEDIUM
    message: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Represents a distributed trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None


class ProductionMetricsCollector:
    """Production-grade metrics collection with real-time monitoring."""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics_buffer = deque(maxlen=max_metrics)
        self.aggregated_metrics = defaultdict(list)
        self.real_time_metrics = {}
        self.metric_callbacks = defaultdict(list)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._aggregation_thread = None
        self.start_aggregation()
    
    def start_aggregation(self):
        """Start background aggregation thread."""
        if self._aggregation_thread is None or not self._aggregation_thread.is_alive():
            self._aggregation_thread = threading.Thread(
                target=self._aggregation_worker,
                daemon=True
            )
            self._aggregation_thread.start()
    
    def stop(self):
        """Stop metrics collection."""
        self._stop_event.set()
        if self._aggregation_thread:
            self._aggregation_thread.join(timeout=5.0)
    
    def record_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric measurement."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics_buffer.append(metric)
            self.real_time_metrics[name] = metric
            
            # Trigger callbacks
            for callback in self.metric_callbacks[name]:
                try:
                    callback(metric)
                except Exception as e:
                    logger.error(f"Metric callback failed for {name}: {e}")
    
    def record_counter(self, name: str, increment: float = 1.0, 
                      tags: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        self.record_metric(name, increment, MetricType.COUNTER, tags)
    
    def record_timer(self, name: str, duration_ms: float,
                    tags: Optional[Dict[str, str]] = None):
        """Record timer metric."""
        self.record_metric(name, duration_ms, MetricType.TIMER, tags)
    
    def get_metric_summary(self, name: str, window_minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get aggregated summary for a metric over time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        values = []
        with self._lock:
            for metric in self.metrics_buffer:
                if metric.name == name and metric.timestamp >= cutoff_time:
                    values.append(metric.value)
        
        if not values:
            return None
        
        return {
            'count': len(values),
            'sum': sum(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'p50': sorted(values)[len(values) // 2],
            'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0],
            'p99': sorted(values)[int(len(values) * 0.99)] if len(values) > 1 else values[0]
        }
    
    def register_metric_callback(self, metric_name: str, callback: Callable[[Metric], None]):
        """Register callback for metric events."""
        self.metric_callbacks[metric_name].append(callback)
    
    def _aggregation_worker(self):
        """Background worker for metric aggregation."""
        while not self._stop_event.is_set():
            try:
                self._aggregate_metrics()
                time.sleep(10)  # Aggregate every 10 seconds
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
    
    def _aggregate_metrics(self):
        """Aggregate metrics for efficient querying."""
        with self._lock:
            # Group metrics by name and time window
            current_time = datetime.now()
            window_start = current_time.replace(second=0, microsecond=0)  # Minute boundary
            
            window_metrics = defaultdict(list)
            
            for metric in self.metrics_buffer:
                if metric.timestamp >= window_start - timedelta(minutes=1):
                    window_metrics[metric.name].append(metric.value)
            
            # Store aggregated values
            for name, values in window_metrics.items():
                if values:
                    self.aggregated_metrics[f"{name}_window"].append({
                        'timestamp': window_start,
                        'count': len(values),
                        'mean': sum(values) / len(values),
                        'max': max(values),
                        'min': min(values)
                    })


class DistributedTracing:
    """Distributed tracing system for request flow monitoring."""
    
    def __init__(self):
        self.active_traces = {}
        self.completed_traces = deque(maxlen=1000)
        self.trace_callbacks = []
        self._lock = threading.RLock()
    
    def start_trace(self, operation_name: str, parent_span_id: Optional[str] = None) -> str:
        """Start a new trace span."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:8]
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name
        )
        
        with self._lock:
            self.active_traces[span_id] = span
        
        logger.debug(f"Started trace span: {operation_name} [{span_id}]")
        return span_id
    
    def finish_trace(self, span_id: str, error: Optional[Exception] = None):
        """Finish a trace span."""
        with self._lock:
            if span_id not in self.active_traces:
                logger.warning(f"Attempted to finish unknown span: {span_id}")
                return
            
            span = self.active_traces.pop(span_id)
            span.end_time = datetime.now()
            
            if error:
                span.error = str(error)
                span.tags['error'] = 'true'
                span.tags['error_type'] = type(error).__name__
            
            self.completed_traces.append(span)
            
            # Notify callbacks
            for callback in self.trace_callbacks:
                try:
                    callback(span)
                except Exception as e:
                    logger.error(f"Trace callback failed: {e}")
        
        logger.debug(f"Finished trace span: {span.operation_name} [{span_id}] "
                    f"({span.duration_ms:.2f}ms)")
    
    def add_span_log(self, span_id: str, message: str, level: str = "INFO"):
        """Add log entry to span."""
        with self._lock:
            if span_id in self.active_traces:
                self.active_traces[span_id].logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'level': level,
                    'message': message
                })
    
    def add_span_tag(self, span_id: str, key: str, value: str):
        """Add tag to span."""
        with self._lock:
            if span_id in self.active_traces:
                self.active_traces[span_id].tags[key] = value
    
    def get_trace_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get trace summary for analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_traces = [
            trace for trace in self.completed_traces
            if trace.start_time >= cutoff_time
        ]
        
        if not recent_traces:
            return {'total_traces': 0}
        
        # Calculate statistics
        durations = [trace.duration_ms for trace in recent_traces if trace.duration_ms]
        error_traces = [trace for trace in recent_traces if trace.error]
        
        operations = defaultdict(list)
        for trace in recent_traces:
            operations[trace.operation_name].append(trace.duration_ms or 0)
        
        return {
            'total_traces': len(recent_traces),
            'error_rate': len(error_traces) / len(recent_traces) if recent_traces else 0,
            'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
            'p95_duration_ms': sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            'operations': {
                name: {
                    'count': len(times),
                    'avg_duration_ms': sum(times) / len(times) if times else 0
                }
                for name, times in operations.items()
            }
        }
    
    def register_trace_callback(self, callback: Callable[[TraceSpan], None]):
        """Register callback for completed traces."""
        self.trace_callbacks.append(callback)


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_rules = []
        self.notification_callbacks = defaultdict(list)
        self._lock = threading.RLock()
    
    def register_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                           severity: AlertSeverity = AlertSeverity.MEDIUM,
                           message_template: str = "Alert triggered: {name}"):
        """Register an alert rule."""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'message_template': message_template,
            'last_triggered': None
        })
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics):
                    # Check if we should trigger (avoid spam)
                    if (rule['last_triggered'] is None or 
                        current_time - rule['last_triggered'] > timedelta(minutes=5)):
                        
                        alert = Alert(
                            severity=rule['severity'],
                            message=rule['message_template'].format(name=rule['name']),
                            source=rule['name'],
                            context={'triggered_metrics': metrics}
                        )
                        
                        self.trigger_alert(alert)
                        rule['last_triggered'] = current_time
                        
            except Exception as e:
                logger.error(f"Alert rule {rule['name']} evaluation failed: {e}")
    
    def trigger_alert(self, alert: Alert):
        """Trigger a new alert."""
        with self._lock:
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
        
        logger.warning(f"ALERT [{alert.severity.value}]: {alert.message}")
        
        # Notify callbacks
        for callback in self.notification_callbacks[alert.severity]:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert notification callback failed: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get currently active alerts."""
        with self._lock:
            alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            return alerts
    
    def register_notification_callback(self, severity: AlertSeverity, callback: Callable[[Alert], None]):
        """Register callback for alert notifications."""
        self.notification_callbacks[severity].append(callback)


class HealthCheckManager:
    """Manages system health checks with intelligent scheduling."""
    
    def __init__(self):
        self.health_checks = {}
        self.health_status = {}
        self.check_history = defaultdict(lambda: deque(maxlen=100))
        self.check_intervals = {}
        self.last_check_times = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._health_thread = None
        self.start_health_monitoring()
    
    def register_health_check(self, name: str, check_func: Callable[[], bool],
                            interval_seconds: int = 30,
                            critical: bool = False):
        """Register a health check function."""
        with self._lock:
            self.health_checks[name] = {
                'func': check_func,
                'critical': critical,
                'enabled': True
            }
            self.check_intervals[name] = interval_seconds
            self.health_status[name] = {'status': 'unknown', 'last_check': None}
    
    def start_health_monitoring(self):
        """Start background health monitoring."""
        if self._health_thread is None or not self._health_thread.is_alive():
            self._health_thread = threading.Thread(
                target=self._health_monitoring_worker,
                daemon=True
            )
            self._health_thread.start()
    
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        self._stop_event.set()
        if self._health_thread:
            self._health_thread.join(timeout=10.0)
    
    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return {'status': 'unknown', 'error': 'Health check not found'}
        
        check_info = self.health_checks[name]
        if not check_info['enabled']:
            return {'status': 'disabled'}
        
        start_time = datetime.now()
        try:
            result = check_info['func']()
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            status = {
                'status': 'healthy' if result else 'unhealthy',
                'last_check': start_time.isoformat(),
                'duration_ms': duration_ms,
                'critical': check_info['critical']
            }
            
            with self._lock:
                self.health_status[name] = status
                self.check_history[name].append(status)
                self.last_check_times[name] = start_time
            
            return status
            
        except Exception as e:
            error_status = {
                'status': 'error',
                'error': str(e),
                'last_check': start_time.isoformat(),
                'critical': check_info['critical']
            }
            
            with self._lock:
                self.health_status[name] = error_status
                self.check_history[name].append(error_status)
            
            logger.error(f"Health check {name} failed: {e}")
            return error_status
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self._lock:
            all_checks = list(self.health_status.values())
            
            if not all_checks:
                return {'overall': 'unknown', 'checks': {}}
            
            healthy_checks = sum(1 for check in all_checks if check.get('status') == 'healthy')
            critical_failures = sum(1 for check in all_checks 
                                  if check.get('critical', False) and check.get('status') != 'healthy')
            
            if critical_failures > 0:
                overall_status = 'critical'
            elif healthy_checks == len(all_checks):
                overall_status = 'healthy'
            elif healthy_checks >= len(all_checks) * 0.8:
                overall_status = 'degraded'
            else:
                overall_status = 'unhealthy'
            
            return {
                'overall': overall_status,
                'checks': self.health_status.copy(),
                'summary': {
                    'total_checks': len(all_checks),
                    'healthy_checks': healthy_checks,
                    'critical_failures': critical_failures
                }
            }
    
    def _health_monitoring_worker(self):
        """Background worker for automated health checks."""
        while not self._stop_event.is_set():
            current_time = datetime.now()
            
            for name, interval in self.check_intervals.items():
                last_check = self.last_check_times.get(name)
                
                if (last_check is None or 
                    current_time - last_check >= timedelta(seconds=interval)):
                    try:
                        self.run_health_check(name)
                    except Exception as e:
                        logger.error(f"Automated health check {name} failed: {e}")
            
            time.sleep(5)  # Check every 5 seconds


class ProductionMonitoringSystem:
    """Comprehensive production monitoring system."""
    
    def __init__(self):
        self.metrics_collector = ProductionMetricsCollector()
        self.tracing = DistributedTracing()
        self.alert_manager = AlertManager()
        self.health_manager = HealthCheckManager()
        self.performance_monitor = PerformanceMonitor()
        
        # System metrics tracking
        self.system_metrics_enabled = True
        self._system_metrics_thread = None
        self._setup_default_alerts()
        self._setup_default_health_checks()
        self.start_system_monitoring()
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High memory usage alert
        self.alert_manager.register_alert_rule(
            "high_memory_usage",
            lambda m: m.get('memory_percent', 0) > 85,
            AlertSeverity.HIGH,
            "High memory usage detected: {memory_percent}%"
        )
        
        # High error rate alert
        self.alert_manager.register_alert_rule(
            "high_error_rate",
            lambda m: m.get('error_rate', 0) > 0.1,
            AlertSeverity.CRITICAL,
            "High error rate detected: {error_rate}"
        )
        
        # Slow response time alert
        self.alert_manager.register_alert_rule(
            "slow_response_time",
            lambda m: m.get('avg_response_time_ms', 0) > 5000,
            AlertSeverity.MEDIUM,
            "Slow response times detected: {avg_response_time_ms}ms"
        )
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        self.health_manager.register_health_check(
            "memory_usage",
            self._check_memory_health,
            interval_seconds=30,
            critical=True
        )
        
        self.health_manager.register_health_check(
            "disk_space",
            self._check_disk_health,
            interval_seconds=60,
            critical=True
        )
        
        self.health_manager.register_health_check(
            "system_load",
            self._check_system_load,
            interval_seconds=15,
            critical=False
        )
    
    def _check_memory_health(self) -> bool:
        """Check system memory health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric('memory_percent', memory.percent)
            return memory.percent < 90
        except ImportError:
            return True
    
    def _check_disk_health(self) -> bool:
        """Check disk space health."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            self.metrics_collector.record_metric('disk_usage_percent', usage_percent)
            return usage_percent < 90
        except ImportError:
            return True
    
    def _check_system_load(self) -> bool:
        """Check system load health."""
        try:
            import psutil
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_metric('cpu_percent', cpu_percent)
            self.metrics_collector.record_metric('load_average', load_avg)
            return cpu_percent < 80
        except ImportError:
            return True
    
    def start_system_monitoring(self):
        """Start system-level metrics monitoring."""
        if self.system_metrics_enabled:
            self._system_metrics_thread = threading.Thread(
                target=self._system_metrics_worker,
                daemon=True
            )
            self._system_metrics_thread.start()
    
    def _system_metrics_worker(self):
        """Background worker for system metrics."""
        while self.system_metrics_enabled:
            try:
                # Collect system metrics
                current_metrics = self._collect_system_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts(current_metrics)
                
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        try:
            import psutil
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available_mb'] = memory.available / (1024 * 1024)
            
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_usage_percent'] = (disk.used / disk.total) * 100
            metrics['disk_free_gb'] = disk.free / (1024 * 1024 * 1024)
            
            # Process metrics
            process = psutil.Process()
            metrics['process_memory_mb'] = process.memory_info().rss / (1024 * 1024)
            metrics['process_cpu_percent'] = process.cpu_percent()
            
        except ImportError:
            # Fallback metrics when psutil is not available
            metrics['system_check'] = 1.0
        
        # Record metrics
        for name, value in metrics.items():
            self.metrics_collector.record_metric(name, value)
        
        return metrics
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'health': self.health_manager.get_overall_health(),
            'alerts': {
                'active': len(self.alert_manager.get_active_alerts()),
                'critical': len(self.alert_manager.get_active_alerts(AlertSeverity.CRITICAL)),
                'recent': len([a for a in self.alert_manager.alert_history 
                             if datetime.now() - a.timestamp <= timedelta(hours=1)])
            },
            'traces': self.tracing.get_trace_summary(),
            'metrics': {
                name: self.metrics_collector.get_metric_summary(name)
                for name in ['memory_percent', 'cpu_percent', 'disk_usage_percent']
                if self.metrics_collector.get_metric_summary(name) is not None
            }
        }
    
    def shutdown(self):
        """Graceful shutdown of monitoring system."""
        logger.info("Shutting down production monitoring system")
        
        self.system_metrics_enabled = False
        self.metrics_collector.stop()
        self.health_manager.stop_health_monitoring()
        
        # Save final monitoring report
        try:
            dashboard = self.get_monitoring_dashboard()
            report_file = f'/root/repo/logs/monitoring_shutdown_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_file, 'w') as f:
                json.dump(dashboard, f, indent=2)
            logger.info(f"Final monitoring report saved: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save shutdown report: {e}")


class PerformanceMonitor:
    """Monitors performance characteristics and detects anomalies."""
    
    def __init__(self):
        self.operation_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            'response_time_multiplier': 3.0,  # 3x baseline
            'error_rate_threshold': 0.05,     # 5% error rate
            'throughput_drop_threshold': 0.5   # 50% throughput drop
        }
        self._lock = threading.RLock()
    
    def record_operation_metrics(self, operation: str, duration_ms: float, 
                               success: bool, context: Optional[Dict] = None):
        """Record metrics for an operation."""
        metric_entry = {
            'timestamp': datetime.now(),
            'duration_ms': duration_ms,
            'success': success,
            'context': context or {}
        }
        
        with self._lock:
            self.operation_metrics[operation].append(metric_entry)
            
            # Update baseline if we have enough data
            if len(self.operation_metrics[operation]) >= 50:
                self._update_baseline(operation)
    
    def _update_baseline(self, operation: str):
        """Update baseline metrics for operation."""
        recent_metrics = list(self.operation_metrics[operation])[-50:]  # Last 50 operations
        successful_metrics = [m for m in recent_metrics if m['success']]
        
        if successful_metrics:
            durations = [m['duration_ms'] for m in successful_metrics]
            self.baseline_metrics[operation] = {
                'avg_duration_ms': sum(durations) / len(durations),
                'p95_duration_ms': sorted(durations)[int(len(durations) * 0.95)],
                'success_rate': len(successful_metrics) / len(recent_metrics)
            }
    
    def detect_anomalies(self, operation: str) -> List[Dict[str, Any]]:
        """Detect performance anomalies for an operation."""
        if operation not in self.baseline_metrics:
            return []
        
        baseline = self.baseline_metrics[operation]
        recent_metrics = list(self.operation_metrics[operation])[-10:]  # Last 10 operations
        
        if not recent_metrics:
            return []
        
        anomalies = []
        
        # Check response time anomalies
        recent_durations = [m['duration_ms'] for m in recent_metrics if m['success']]
        if recent_durations:
            avg_recent = sum(recent_durations) / len(recent_durations)
            if avg_recent > baseline['avg_duration_ms'] * self.anomaly_thresholds['response_time_multiplier']:
                anomalies.append({
                    'type': 'slow_response',
                    'severity': 'HIGH',
                    'current': avg_recent,
                    'baseline': baseline['avg_duration_ms']
                })
        
        # Check error rate anomalies
        recent_error_rate = 1 - (sum(1 for m in recent_metrics if m['success']) / len(recent_metrics))
        if recent_error_rate > self.anomaly_thresholds['error_rate_threshold']:
            anomalies.append({
                'type': 'high_error_rate',
                'severity': 'CRITICAL',
                'current': recent_error_rate,
                'threshold': self.anomaly_thresholds['error_rate_threshold']
            })
        
        return anomalies


# Context managers for easy monitoring integration
@contextmanager
def monitor_operation(operation_name: str, monitoring_system: Optional[ProductionMonitoringSystem] = None):
    """Context manager for monitoring operations with distributed tracing."""
    if monitoring_system is None:
        monitoring_system = global_monitoring_system
    
    span_id = monitoring_system.tracing.start_trace(operation_name)
    start_time = time.time()
    error = None
    
    try:
        yield span_id
    except Exception as e:
        error = e
        monitoring_system.tracing.add_span_log(span_id, f"Error: {str(e)}", "ERROR")
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        success = error is None
        
        monitoring_system.tracing.finish_trace(span_id, error)
        monitoring_system.performance_monitor.record_operation_metrics(
            operation_name, duration_ms, success
        )
        monitoring_system.metrics_collector.record_timer(
            f"operation_duration_{operation_name}", duration_ms
        )


def monitored_function(operation_name: Optional[str] = None):
    """Decorator to add comprehensive monitoring to functions."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with monitor_operation(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def async_monitored_function(operation_name: Optional[str] = None):
    """Decorator to add comprehensive monitoring to async functions."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with monitor_operation(op_name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global monitoring system instance
global_monitoring_system = ProductionMonitoringSystem()


# Production-ready health check functions
def check_torch_gpu_health() -> bool:
    """Check PyTorch GPU health."""
    try:
        import torch
        if torch.cuda.is_available():
            # Test basic GPU operation
            x = torch.randn(10, 10).cuda()
            y = x @ x.T
            return not torch.isnan(y).any().item()
        return True  # No GPU, but that's fine
    except Exception:
        return False


def check_model_loading_health() -> bool:
    """Check if models can be loaded properly."""
    try:
        # Test basic model creation
        import torch.nn as nn
        model = nn.Linear(10, 1)
        x = torch.randn(1, 10)
        output = model(x)
        return not torch.isnan(output).any().item()
    except Exception:
        return False


def check_logging_health() -> bool:
    """Check logging system health."""
    try:
        test_logger = logging.getLogger('health_check_test')
        test_logger.info("Health check test")
        return True
    except Exception:
        return False


# Register production health checks
global_monitoring_system.health_manager.register_health_check(
    "torch_gpu", check_torch_gpu_health, interval_seconds=60
)
global_monitoring_system.health_manager.register_health_check(
    "model_loading", check_model_loading_health, interval_seconds=120
)
global_monitoring_system.health_manager.register_health_check(
    "logging_system", check_logging_health, interval_seconds=300
)


# Graceful shutdown handling
def setup_graceful_shutdown():
    """Setup graceful shutdown handlers."""
    import signal
    import atexit
    
    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        global_monitoring_system.shutdown()
    
    def cleanup_on_exit():
        global_monitoring_system.shutdown()
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    # Register cleanup on normal exit
    atexit.register(cleanup_on_exit)


# Initialize graceful shutdown on module import
setup_graceful_shutdown()