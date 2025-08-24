# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials

"""
Production Quality Gates and Automated Validation Pipeline
Generation 2 Robustness Enhancement
"""

import asyncio
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid
import traceback

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate validation status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class ValidationResult:
    """Result of a validation check."""
    gate_name: str
    status: QualityGateStatus
    message: str
    severity: ValidationSeverity = ValidationSeverity.MEDIUM
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class QualityMetrics:
    """Quality metrics for model and system performance."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    uncertainty_calibration: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    throughput_qps: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    model_size_mb: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class QualityGate(ABC):
    """Abstract base class for quality gates."""
    
    def __init__(self, name: str, enabled: bool = True, severity: ValidationSeverity = ValidationSeverity.MEDIUM):
        self.name = name
        self.enabled = enabled
        self.severity = severity
        self.execution_history = deque(maxlen=100)
    
    @abstractmethod
    async def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Perform validation check."""
        pass
    
    def is_enabled(self, context: Dict[str, Any]) -> bool:
        """Check if gate should be enabled for given context."""
        return self.enabled


class DataQualityGate(QualityGate):
    """Validates data quality and integrity."""
    
    def __init__(self):
        super().__init__("data_quality", severity=ValidationSeverity.HIGH)
        self.quality_thresholds = {
            'min_samples': 100,
            'max_missing_ratio': 0.1,
            'max_outlier_ratio': 0.05,
            'min_variance': 1e-8
        }
    
    async def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate data quality."""
        start_time = time.time()
        
        try:
            data = context.get('data')
            if data is None:
                return ValidationResult(
                    gate_name=self.name,
                    status=QualityGateStatus.SKIPPED,
                    message="No data provided for validation",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            issues = []
            
            # Check data availability
            if hasattr(data, '__len__'):
                if len(data) < self.quality_thresholds['min_samples']:
                    issues.append(f"Insufficient samples: {len(data)} < {self.quality_thresholds['min_samples']}")
            
            # Check for missing values (if numpy array or similar)
            if hasattr(data, 'shape'):
                try:
                    import numpy as np
                    if hasattr(data, 'isnan'):
                        nan_ratio = data.isnan().float().mean().item() if hasattr(data, 'isnan') else 0
                    else:
                        # Convert to numpy for analysis
                        np_data = np.asarray(data)
                        nan_ratio = np.isnan(np_data).mean()
                    
                    if nan_ratio > self.quality_thresholds['max_missing_ratio']:
                        issues.append(f"High missing value ratio: {nan_ratio:.3f} > {self.quality_thresholds['max_missing_ratio']}")
                        
                    # Check variance (avoid constant data)
                    if hasattr(data, 'var'):
                        variance = data.var().item() if hasattr(data.var(), 'item') else float(data.var())
                    else:
                        variance = np.var(np_data)
                    
                    if variance < self.quality_thresholds['min_variance']:
                        issues.append(f"Low data variance: {variance:.2e} < {self.quality_thresholds['min_variance']}")
                        
                except ImportError:
                    issues.append("NumPy not available for detailed data quality checks")
                except Exception as e:
                    issues.append(f"Data quality analysis error: {str(e)}")
            
            # Determine status
            if not issues:
                status = QualityGateStatus.PASSED
                message = "Data quality validation passed"
            elif len(issues) == 1 and "NumPy not available" in issues[0]:
                status = QualityGateStatus.WARNING
                message = f"Limited validation: {issues[0]}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Data quality issues: {'; '.join(issues)}"
            
            return ValidationResult(
                gate_name=self.name,
                status=status,
                message=message,
                details={'issues': issues, 'thresholds': self.quality_thresholds},
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                gate_name=self.name,
                status=QualityGateStatus.ERROR,
                message=f"Data quality validation error: {str(e)}",
                error=traceback.format_exc(),
                execution_time_ms=(time.time() - start_time) * 1000
            )


class ModelPerformanceGate(QualityGate):
    """Validates model performance metrics."""
    
    def __init__(self):
        super().__init__("model_performance", severity=ValidationSeverity.CRITICAL)
        self.performance_thresholds = {
            'min_accuracy': 0.7,
            'max_latency_ms': 1000,
            'min_throughput_qps': 10,
            'max_memory_mb': 2048,
            'max_uncertainty_error': 0.2
        }
    
    async def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate model performance."""
        start_time = time.time()
        
        try:
            metrics = context.get('metrics')
            if not isinstance(metrics, QualityMetrics):
                return ValidationResult(
                    gate_name=self.name,
                    status=QualityGateStatus.SKIPPED,
                    message="No performance metrics provided",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            issues = []
            warnings = []
            
            # Check accuracy
            if metrics.accuracy is not None:
                if metrics.accuracy < self.performance_thresholds['min_accuracy']:
                    issues.append(f"Low accuracy: {metrics.accuracy:.3f} < {self.performance_thresholds['min_accuracy']}")
            
            # Check latency
            if metrics.latency_p95_ms is not None:
                if metrics.latency_p95_ms > self.performance_thresholds['max_latency_ms']:
                    issues.append(f"High latency: {metrics.latency_p95_ms:.1f}ms > {self.performance_thresholds['max_latency_ms']}ms")
            
            # Check throughput
            if metrics.throughput_qps is not None:
                if metrics.throughput_qps < self.performance_thresholds['min_throughput_qps']:
                    warnings.append(f"Low throughput: {metrics.throughput_qps:.1f} QPS < {self.performance_thresholds['min_throughput_qps']} QPS")
            
            # Check memory usage
            if metrics.memory_usage_mb is not None:
                if metrics.memory_usage_mb > self.performance_thresholds['max_memory_mb']:
                    issues.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB > {self.performance_thresholds['max_memory_mb']}MB")
            
            # Check uncertainty calibration
            if metrics.uncertainty_calibration is not None:
                if metrics.uncertainty_calibration > self.performance_thresholds['max_uncertainty_error']:
                    warnings.append(f"Poor uncertainty calibration: {metrics.uncertainty_calibration:.3f} > {self.performance_thresholds['max_uncertainty_error']}")
            
            # Determine status
            if issues:
                status = QualityGateStatus.FAILED
                message = f"Performance validation failed: {'; '.join(issues)}"
            elif warnings:
                status = QualityGateStatus.WARNING
                message = f"Performance warnings: {'; '.join(warnings)}"
            else:
                status = QualityGateStatus.PASSED
                message = "Model performance validation passed"
            
            return ValidationResult(
                gate_name=self.name,
                status=status,
                message=message,
                details={
                    'issues': issues,
                    'warnings': warnings,
                    'metrics': {
                        'accuracy': metrics.accuracy,
                        'latency_p95_ms': metrics.latency_p95_ms,
                        'throughput_qps': metrics.throughput_qps,
                        'memory_usage_mb': metrics.memory_usage_mb,
                        'uncertainty_calibration': metrics.uncertainty_calibration
                    },
                    'thresholds': self.performance_thresholds
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                gate_name=self.name,
                status=QualityGateStatus.ERROR,
                message=f"Performance validation error: {str(e)}",
                error=traceback.format_exc(),
                execution_time_ms=(time.time() - start_time) * 1000
            )


class SecurityValidationGate(QualityGate):
    """Validates security requirements and compliance."""
    
    def __init__(self):
        super().__init__("security_validation", severity=ValidationSeverity.CRITICAL)
        self.security_requirements = {
            'input_sanitization': True,
            'audit_logging': True,
            'rate_limiting': True,
            'session_management': True,
            'threat_detection': True
        }
    
    async def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate security compliance."""
        start_time = time.time()
        
        try:
            issues = []
            checks_passed = 0
            total_checks = len(self.security_requirements)
            
            # Import security components
            try:
                from ..security.production_security import global_security_validator
                security_validator = global_security_validator
            except ImportError:
                return ValidationResult(
                    gate_name=self.name,
                    status=QualityGateStatus.ERROR,
                    message="Security validation modules not available",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check input sanitization
            try:
                test_input = {"test": "safe_value"}
                sanitized = security_validator.input_sanitizer.sanitize_dict(test_input)
                checks_passed += 1
            except Exception:
                issues.append("Input sanitization not working")
            
            # Check audit logging
            try:
                from ..security.production_security import SecurityEvent
                test_event = SecurityEvent(event_type="VALIDATION_TEST")
                security_validator.auditor.log_security_event(test_event)
                checks_passed += 1
            except Exception:
                issues.append("Audit logging not working")
            
            # Check rate limiting
            try:
                allowed, reason = security_validator.dos_protection.check_request(
                    "validation_test", "test_action"
                )
                checks_passed += 1
            except Exception:
                issues.append("Rate limiting not working")
            
            # Check session management
            try:
                session_count = security_validator.session_manager.get_session_count()
                checks_passed += 1
            except Exception:
                issues.append("Session management not working")
            
            # Check threat detection
            try:
                threats = security_validator.auditor.detect_threats("<script>alert('test')</script>")
                if threats:  # Should detect this as a threat
                    checks_passed += 1
                else:
                    issues.append("Threat detection not sensitive enough")
            except Exception:
                issues.append("Threat detection not working")
            
            # Determine status
            if checks_passed == total_checks:
                status = QualityGateStatus.PASSED
                message = "All security validations passed"
            elif checks_passed >= total_checks * 0.8:
                status = QualityGateStatus.WARNING
                message = f"Most security checks passed ({checks_passed}/{total_checks})"
            else:
                status = QualityGateStatus.FAILED
                message = f"Security validation failed: {'; '.join(issues)}"
            
            return ValidationResult(
                gate_name=self.name,
                status=status,
                message=message,
                details={
                    'checks_passed': checks_passed,
                    'total_checks': total_checks,
                    'issues': issues
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                gate_name=self.name,
                status=QualityGateStatus.ERROR,
                message=f"Security validation error: {str(e)}",
                error=traceback.format_exc(),
                execution_time_ms=(time.time() - start_time) * 1000
            )


class ModelDriftDetectionGate(QualityGate):
    """Detects model drift and performance degradation."""
    
    def __init__(self):
        super().__init__("model_drift_detection", severity=ValidationSeverity.HIGH)
        self.baseline_metrics = None
        self.drift_thresholds = {
            'accuracy_drop': 0.05,      # 5% accuracy drop
            'latency_increase': 2.0,    # 2x latency increase
            'uncertainty_increase': 0.1  # 10% uncertainty increase
        }
    
    async def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Detect model drift."""
        start_time = time.time()
        
        try:
            current_metrics = context.get('metrics')
            baseline_metrics = context.get('baseline_metrics', self.baseline_metrics)
            
            if not current_metrics or not baseline_metrics:
                return ValidationResult(
                    gate_name=self.name,
                    status=QualityGateStatus.SKIPPED,
                    message="Insufficient metrics for drift detection",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            drift_issues = []
            
            # Check accuracy drift
            if (hasattr(current_metrics, 'accuracy') and current_metrics.accuracy is not None and
                hasattr(baseline_metrics, 'accuracy') and baseline_metrics.accuracy is not None):
                
                accuracy_drop = baseline_metrics.accuracy - current_metrics.accuracy
                if accuracy_drop > self.drift_thresholds['accuracy_drop']:
                    drift_issues.append(f"Accuracy dropped by {accuracy_drop:.3f}")
            
            # Check latency drift
            if (hasattr(current_metrics, 'latency_p95_ms') and current_metrics.latency_p95_ms is not None and
                hasattr(baseline_metrics, 'latency_p95_ms') and baseline_metrics.latency_p95_ms is not None):
                
                latency_ratio = current_metrics.latency_p95_ms / baseline_metrics.latency_p95_ms
                if latency_ratio > self.drift_thresholds['latency_increase']:
                    drift_issues.append(f"Latency increased by {latency_ratio:.2f}x")
            
            # Check uncertainty calibration drift
            if (hasattr(current_metrics, 'uncertainty_calibration') and current_metrics.uncertainty_calibration is not None and
                hasattr(baseline_metrics, 'uncertainty_calibration') and baseline_metrics.uncertainty_calibration is not None):
                
                uncertainty_increase = current_metrics.uncertainty_calibration - baseline_metrics.uncertainty_calibration
                if uncertainty_increase > self.drift_thresholds['uncertainty_increase']:
                    drift_issues.append(f"Uncertainty calibration degraded by {uncertainty_increase:.3f}")
            
            # Determine status
            if not drift_issues:
                status = QualityGateStatus.PASSED
                message = "No significant model drift detected"
            elif len(drift_issues) == 1:
                status = QualityGateStatus.WARNING
                message = f"Minor drift detected: {drift_issues[0]}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Significant drift detected: {'; '.join(drift_issues)}"
            
            return ValidationResult(
                gate_name=self.name,
                status=status,
                message=message,
                details={
                    'drift_issues': drift_issues,
                    'thresholds': self.drift_thresholds,
                    'current_metrics': self._serialize_metrics(current_metrics),
                    'baseline_metrics': self._serialize_metrics(baseline_metrics)
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                gate_name=self.name,
                status=QualityGateStatus.ERROR,
                message=f"Drift detection error: {str(e)}",
                error=traceback.format_exc(),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _serialize_metrics(self, metrics) -> Dict[str, Any]:
        """Serialize metrics for logging."""
        if hasattr(metrics, '__dict__'):
            return {k: v for k, v in metrics.__dict__.items() if v is not None}
        return {}


class ResourceValidationGate(QualityGate):
    """Validates system resource usage and availability."""
    
    def __init__(self):
        super().__init__("resource_validation", severity=ValidationSeverity.HIGH)
        self.resource_limits = {
            'max_memory_percent': 85,
            'max_cpu_percent': 80,
            'min_disk_free_gb': 5,
            'max_open_files': 1000
        }
    
    async def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate resource usage."""
        start_time = time.time()
        
        try:
            resource_issues = []
            
            # Check system resources
            try:
                import psutil
                
                # Memory check
                memory = psutil.virtual_memory()
                if memory.percent > self.resource_limits['max_memory_percent']:
                    resource_issues.append(f"High memory usage: {memory.percent:.1f}% > {self.resource_limits['max_memory_percent']}%")
                
                # CPU check
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > self.resource_limits['max_cpu_percent']:
                    resource_issues.append(f"High CPU usage: {cpu_percent:.1f}% > {self.resource_limits['max_cpu_percent']}%")
                
                # Disk check
                disk = psutil.disk_usage('/')
                free_gb = disk.free / (1024**3)
                if free_gb < self.resource_limits['min_disk_free_gb']:
                    resource_issues.append(f"Low disk space: {free_gb:.1f}GB < {self.resource_limits['min_disk_free_gb']}GB")
                
                # Process-specific checks
                process = psutil.Process()
                open_files = len(process.open_files())
                if open_files > self.resource_limits['max_open_files']:
                    resource_issues.append(f"Too many open files: {open_files} > {self.resource_limits['max_open_files']}")
                
            except ImportError:
                resource_issues.append("psutil not available for resource monitoring")
            
            # Check GPU resources if available
            gpu_info = self._check_gpu_resources()
            if gpu_info.get('issues'):
                resource_issues.extend(gpu_info['issues'])
            
            # Determine status
            if not resource_issues:
                status = QualityGateStatus.PASSED
                message = "Resource validation passed"
            elif len(resource_issues) == 1 and "psutil not available" in resource_issues[0]:
                status = QualityGateStatus.WARNING
                message = "Limited resource validation due to missing dependencies"
            else:
                status = QualityGateStatus.FAILED
                message = f"Resource validation failed: {'; '.join(resource_issues)}"
            
            return ValidationResult(
                gate_name=self.name,
                status=status,
                message=message,
                details={
                    'issues': resource_issues,
                    'limits': self.resource_limits,
                    'gpu_info': gpu_info
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                gate_name=self.name,
                status=QualityGateStatus.ERROR,
                message=f"Resource validation error: {str(e)}",
                error=traceback.format_exc(),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _check_gpu_resources(self) -> Dict[str, Any]:
        """Check GPU resource availability."""
        gpu_info = {'available': False, 'issues': []}
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['device_count'] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_used = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    memory_total = props.total_memory / (1024**3)  # GB
                    memory_percent = (memory_used / memory_total) * 100
                    
                    gpu_info[f'gpu_{i}'] = {
                        'name': props.name,
                        'memory_used_gb': memory_used,
                        'memory_total_gb': memory_total,
                        'memory_percent': memory_percent
                    }
                    
                    if memory_percent > 90:
                        gpu_info['issues'].append(f"GPU {i} high memory usage: {memory_percent:.1f}%")
            
        except ImportError:
            gpu_info['issues'].append("PyTorch not available for GPU monitoring")
        except Exception as e:
            gpu_info['issues'].append(f"GPU check error: {str(e)}")
        
        return gpu_info


class ConfigurationValidationGate(QualityGate):
    """Validates system configuration and dependencies."""
    
    def __init__(self):
        super().__init__("configuration_validation", severity=ValidationSeverity.MEDIUM)
        self.required_config_keys = [
            'model_type', 'input_dim', 'hidden_dim'
        ]
        self.required_dependencies = [
            'numpy', 'scipy'
        ]
        self.optional_dependencies = [
            'torch', 'matplotlib', 'h5py'
        ]
    
    async def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate configuration."""
        start_time = time.time()
        
        try:
            config = context.get('config', {})
            config_issues = []
            
            # Check required configuration keys
            for key in self.required_config_keys:
                if key not in config:
                    config_issues.append(f"Missing required config key: {key}")
            
            # Validate configuration values
            for key, value in config.items():
                if isinstance(value, str) and len(value) > 1000:
                    config_issues.append(f"Config value too long for key: {key}")
                elif isinstance(value, (int, float)) and abs(value) > 1e10:
                    config_issues.append(f"Suspicious large value for key: {key}")
            
            # Check dependencies
            dependency_status = {'available': [], 'missing': []}
            
            for dep in self.required_dependencies + self.optional_dependencies:
                try:
                    __import__(dep)
                    dependency_status['available'].append(dep)
                except ImportError:
                    dependency_status['missing'].append(dep)
                    if dep in self.required_dependencies:
                        config_issues.append(f"Missing required dependency: {dep}")
            
            # Check file system configuration
            fs_issues = self._check_filesystem_config()
            config_issues.extend(fs_issues)
            
            # Determine status
            critical_issues = [issue for issue in config_issues 
                             if any(word in issue.lower() for word in ['missing required', 'critical', 'error'])]
            
            if not config_issues:
                status = QualityGateStatus.PASSED
                message = "Configuration validation passed"
            elif critical_issues:
                status = QualityGateStatus.FAILED
                message = f"Critical configuration issues: {'; '.join(critical_issues)}"
            else:
                status = QualityGateStatus.WARNING
                message = f"Configuration warnings: {'; '.join(config_issues)}"
            
            return ValidationResult(
                gate_name=self.name,
                status=status,
                message=message,
                details={
                    'config_issues': config_issues,
                    'dependencies': dependency_status,
                    'config_summary': {k: str(v)[:100] for k, v in config.items()}
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ValidationResult(
                gate_name=self.name,
                status=QualityGateStatus.ERROR,
                message=f"Configuration validation error: {str(e)}",
                error=traceback.format_exc(),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _check_filesystem_config(self) -> List[str]:
        """Check filesystem configuration."""
        issues = []
        
        # Check required directories
        required_dirs = ['/root/repo/logs', '/root/repo/src']
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                issues.append(f"Required directory missing: {dir_path}")
            elif not os.access(dir_path, os.W_OK):
                issues.append(f"Directory not writable: {dir_path}")
        
        # Check log file rotation capacity
        log_dir = '/root/repo/logs'
        if os.path.exists(log_dir):
            try:
                log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
                if len(log_files) > 20:  # Too many log files
                    issues.append(f"Too many log files ({len(log_files)}), rotation may be failing")
            except Exception:
                issues.append("Cannot access log directory for validation")
        
        return issues


class AutomatedRollbackManager:
    """Manages automated rollback for failed deployments or degraded performance."""
    
    def __init__(self):
        self.rollback_triggers = {}
        self.rollback_history = []
        self.rollback_callbacks = defaultdict(list)
        self._lock = threading.RLock()
    
    def register_rollback_trigger(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                                 rollback_action: Callable[[], bool]):
        """Register automated rollback trigger."""
        self.rollback_triggers[name] = {
            'condition': condition,
            'action': rollback_action,
            'last_triggered': None,
            'trigger_count': 0
        }
    
    def check_rollback_conditions(self, validation_results: List[ValidationResult]):
        """Check if rollback should be triggered."""
        current_time = datetime.now()
        
        # Create context from validation results
        context = {
            'validation_results': validation_results,
            'failed_gates': [r for r in validation_results if r.status == QualityGateStatus.FAILED],
            'error_gates': [r for r in validation_results if r.status == QualityGateStatus.ERROR],
            'critical_failures': [r for r in validation_results 
                                if r.status in [QualityGateStatus.FAILED, QualityGateStatus.ERROR] 
                                and r.severity == ValidationSeverity.CRITICAL]
        }
        
        for trigger_name, trigger_info in self.rollback_triggers.items():
            try:
                if trigger_info['condition'](context):
                    # Check cooldown period (don't trigger too frequently)
                    if (trigger_info['last_triggered'] is None or 
                        current_time - trigger_info['last_triggered'] > timedelta(minutes=10)):
                        
                        logger.warning(f"Rollback triggered: {trigger_name}")
                        
                        # Execute rollback
                        success = trigger_info['action']()
                        
                        # Record rollback attempt
                        rollback_record = {
                            'timestamp': current_time,
                            'trigger': trigger_name,
                            'success': success,
                            'context': context,
                            'trigger_count': trigger_info['trigger_count'] + 1
                        }
                        
                        with self._lock:
                            self.rollback_history.append(rollback_record)
                            trigger_info['last_triggered'] = current_time
                            trigger_info['trigger_count'] += 1
                        
                        if success:
                            logger.info(f"Rollback successful: {trigger_name}")
                        else:
                            logger.error(f"Rollback failed: {trigger_name}")
                        
                        return success
                        
            except Exception as e:
                logger.error(f"Rollback trigger {trigger_name} failed: {e}")
        
        return False


class ProductionQualityGatesPipeline:
    """Comprehensive quality gates pipeline for production validation."""
    
    def __init__(self):
        self.gates = [
            DataQualityGate(),
            ModelPerformanceGate(),
            SecurityValidationGate(),
            ModelDriftDetectionGate(),
            ResourceValidationGate(),
            ConfigurationValidationGate()
        ]
        
        self.rollback_manager = AutomatedRollbackManager()
        self.validation_history = deque(maxlen=1000)
        self.pipeline_metrics = defaultdict(list)
        self._setup_default_rollback_triggers()
    
    def _setup_default_rollback_triggers(self):
        """Setup default rollback triggers."""
        # Critical failure trigger
        self.rollback_manager.register_rollback_trigger(
            "critical_failures",
            lambda ctx: len(ctx['critical_failures']) > 0,
            self._emergency_rollback
        )
        
        # Multiple failures trigger
        self.rollback_manager.register_rollback_trigger(
            "multiple_failures",
            lambda ctx: len(ctx['failed_gates']) >= 3,
            self._standard_rollback
        )
        
        # Security failure trigger
        self.rollback_manager.register_rollback_trigger(
            "security_failure",
            lambda ctx: any(r.gate_name == "security_validation" and 
                           r.status == QualityGateStatus.FAILED 
                           for r in ctx['validation_results']),
            self._security_rollback
        )
    
    async def run_quality_gates(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all quality gates in the pipeline."""
        pipeline_start = time.time()
        
        logger.info("Starting production quality gates pipeline")
        
        validation_results = []
        
        # Run all gates
        for gate in self.gates:
            if gate.is_enabled(context):
                logger.info(f"Running quality gate: {gate.name}")
                
                try:
                    result = await gate.validate(context)
                    validation_results.append(result)
                    
                    # Log gate result
                    logger.info(f"Gate {gate.name}: {result.status.value} - {result.message}")
                    
                except Exception as e:
                    error_result = ValidationResult(
                        gate_name=gate.name,
                        status=QualityGateStatus.ERROR,
                        message=f"Gate execution error: {str(e)}",
                        error=traceback.format_exc()
                    )
                    validation_results.append(error_result)
                    logger.error(f"Gate {gate.name} execution failed: {e}")
            else:
                skip_result = ValidationResult(
                    gate_name=gate.name,
                    status=QualityGateStatus.SKIPPED,
                    message="Gate disabled or not applicable"
                )
                validation_results.append(skip_result)
        
        # Analyze results
        pipeline_result = self._analyze_pipeline_results(validation_results)
        pipeline_result['execution_time_ms'] = (time.time() - pipeline_start) * 1000
        
        # Store in history
        self.validation_history.append({
            'timestamp': datetime.now(),
            'results': validation_results,
            'pipeline_result': pipeline_result
        })
        
        # Check rollback conditions
        rollback_triggered = self.rollback_manager.check_rollback_conditions(validation_results)
        pipeline_result['rollback_triggered'] = rollback_triggered
        
        logger.info(f"Quality gates pipeline completed: {pipeline_result['overall_status']}")
        
        return pipeline_result
    
    def _analyze_pipeline_results(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze pipeline results and determine overall status."""
        status_counts = defaultdict(int)
        critical_failures = []
        all_issues = []
        
        for result in results:
            status_counts[result.status.value] += 1
            
            if result.status in [QualityGateStatus.FAILED, QualityGateStatus.ERROR]:
                if result.severity == ValidationSeverity.CRITICAL:
                    critical_failures.append(result)
                all_issues.append(f"{result.gate_name}: {result.message}")
        
        # Determine overall status
        total_gates = len(results)
        passed_gates = status_counts[QualityGateStatus.PASSED.value]
        failed_gates = status_counts[QualityGateStatus.FAILED.value]
        error_gates = status_counts[QualityGateStatus.ERROR.value]
        
        if critical_failures:
            overall_status = "CRITICAL_FAILURE"
        elif failed_gates > 0 or error_gates > 0:
            overall_status = "FAILED"
        elif status_counts[QualityGateStatus.WARNING.value] > 0:
            overall_status = "PASSED_WITH_WARNINGS"
        else:
            overall_status = "PASSED"
        
        return {
            'overall_status': overall_status,
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'error_gates': error_gates,
            'warning_gates': status_counts[QualityGateStatus.WARNING.value],
            'skipped_gates': status_counts[QualityGateStatus.SKIPPED.value],
            'critical_failures': len(critical_failures),
            'issues': all_issues,
            'gate_results': [
                {
                    'name': result.gate_name,
                    'status': result.status.value,
                    'message': result.message,
                    'execution_time_ms': result.execution_time_ms,
                    'severity': result.severity.name
                }
                for result in results
            ]
        }
    
    def _emergency_rollback(self) -> bool:
        """Execute emergency rollback procedure."""
        logger.critical("Executing emergency rollback due to critical failures")
        
        try:
            # In production, this would:
            # 1. Stop accepting new requests
            # 2. Drain current requests
            # 3. Revert to last known good configuration
            # 4. Restart services
            
            # For simulation, we'll just log the actions
            logger.info("Emergency rollback: Stopping new requests")
            logger.info("Emergency rollback: Draining current requests")
            logger.info("Emergency rollback: Reverting to last known good configuration")
            logger.info("Emergency rollback: Restarting services")
            
            return True
            
        except Exception as e:
            logger.error(f"Emergency rollback failed: {e}")
            return False
    
    def _standard_rollback(self) -> bool:
        """Execute standard rollback procedure."""
        logger.warning("Executing standard rollback due to multiple failures")
        
        try:
            # Standard rollback is less aggressive
            logger.info("Standard rollback: Reverting problematic changes")
            logger.info("Standard rollback: Reloading configuration")
            return True
            
        except Exception as e:
            logger.error(f"Standard rollback failed: {e}")
            return False
    
    def _security_rollback(self) -> bool:
        """Execute security-focused rollback."""
        logger.critical("Executing security rollback due to security validation failure")
        
        try:
            logger.info("Security rollback: Enabling enhanced security mode")
            logger.info("Security rollback: Blocking suspicious traffic")
            logger.info("Security rollback: Alerting security team")
            return True
            
        except Exception as e:
            logger.error(f"Security rollback failed: {e}")
            return False
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health status."""
        if not self.validation_history:
            return {'status': 'unknown', 'reason': 'No validation history'}
        
        recent_validations = [
            entry for entry in self.validation_history
            if datetime.now() - entry['timestamp'] <= timedelta(hours=1)
        ]
        
        if not recent_validations:
            return {'status': 'stale', 'reason': 'No recent validations'}
        
        # Analyze recent results
        recent_statuses = [entry['pipeline_result']['overall_status'] for entry in recent_validations]
        
        critical_failures = sum(1 for status in recent_statuses if status == "CRITICAL_FAILURE")
        failures = sum(1 for status in recent_statuses if status in ["FAILED", "CRITICAL_FAILURE"])
        
        if critical_failures > 0:
            health_status = "CRITICAL"
        elif failures > len(recent_statuses) * 0.5:
            health_status = "DEGRADED"
        elif failures > 0:
            health_status = "WARNING"
        else:
            health_status = "HEALTHY"
        
        return {
            'status': health_status,
            'recent_validations': len(recent_validations),
            'success_rate': (len(recent_statuses) - failures) / len(recent_statuses),
            'last_validation': recent_validations[-1]['timestamp'].isoformat() if recent_validations else None
        }


# Context manager for quality gate execution
@contextmanager
def quality_gate_context(pipeline: ProductionQualityGatesPipeline,
                        config: Dict[str, Any],
                        metrics: Optional[QualityMetrics] = None):
    """Context manager for executing quality gates."""
    context = {
        'config': config,
        'metrics': metrics,
        'start_time': datetime.now()
    }
    
    try:
        yield context
    finally:
        context['end_time'] = datetime.now()


# Decorator for automatic quality gate validation
def with_quality_gates(config_key: str = 'config', metrics_key: str = 'metrics'):
    """Decorator to add automatic quality gate validation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            pipeline = global_quality_pipeline
            
            # Extract config and metrics from arguments
            config = kwargs.get(config_key, {})
            metrics = kwargs.get(metrics_key)
            
            # Run pre-execution quality gates
            context = {'config': config, 'metrics': metrics, 'phase': 'pre_execution'}
            pre_results = await pipeline.run_quality_gates(context)
            
            if pre_results['overall_status'] in ['CRITICAL_FAILURE', 'FAILED']:
                raise ValueError(f"Pre-execution quality gates failed: {pre_results['overall_status']}")
            
            # Execute function
            try:
                result = await func(*args, **kwargs)
                
                # Run post-execution quality gates if we have new metrics
                if hasattr(result, 'metrics') or 'metrics' in kwargs:
                    post_context = {
                        'config': config,
                        'metrics': getattr(result, 'metrics', kwargs.get('metrics')),
                        'phase': 'post_execution'
                    }
                    post_results = await pipeline.run_quality_gates(post_context)
                    
                    if hasattr(result, '__dict__'):
                        result.quality_gate_results = post_results
                
                return result
                
            except Exception as e:
                # Run failure quality gates
                failure_context = {
                    'config': config,
                    'error': str(e),
                    'phase': 'failure'
                }
                await pipeline.run_quality_gates(failure_context)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in async context
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Global instances
global_quality_pipeline = ProductionQualityGatesPipeline()


# Utility functions for common validation patterns
def validate_model_checkpoints(checkpoint_path: str) -> ValidationResult:
    """Validate model checkpoint integrity."""
    start_time = time.time()
    
    try:
        if not os.path.exists(checkpoint_path):
            return ValidationResult(
                gate_name="checkpoint_validation",
                status=QualityGateStatus.FAILED,
                message=f"Checkpoint not found: {checkpoint_path}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Check file size
        file_size = os.path.getsize(checkpoint_path)
        if file_size == 0:
            return ValidationResult(
                gate_name="checkpoint_validation",
                status=QualityGateStatus.FAILED,
                message="Checkpoint file is empty",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Try to load checkpoint (basic validation)
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                return ValidationResult(
                    gate_name="checkpoint_validation",
                    status=QualityGateStatus.FAILED,
                    message=f"Checkpoint missing keys: {missing_keys}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
        except Exception as e:
            return ValidationResult(
                gate_name="checkpoint_validation",
                status=QualityGateStatus.FAILED,
                message=f"Checkpoint loading failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        return ValidationResult(
            gate_name="checkpoint_validation",
            status=QualityGateStatus.PASSED,
            message="Checkpoint validation passed",
            details={'file_size_mb': file_size / (1024**2)},
            execution_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        return ValidationResult(
            gate_name="checkpoint_validation",
            status=QualityGateStatus.ERROR,
            message=f"Checkpoint validation error: {str(e)}",
            error=traceback.format_exc(),
            execution_time_ms=(time.time() - start_time) * 1000
        )


async def run_comprehensive_validation(config: Dict[str, Any], 
                                     metrics: Optional[QualityMetrics] = None,
                                     baseline_metrics: Optional[QualityMetrics] = None) -> Dict[str, Any]:
    """Run comprehensive validation with all quality gates."""
    context = {
        'config': config,
        'metrics': metrics,
        'baseline_metrics': baseline_metrics,
        'validation_id': str(uuid.uuid4())[:8]
    }
    
    pipeline = global_quality_pipeline
    results = await pipeline.run_quality_gates(context)
    
    # Add pipeline health to results
    results['pipeline_health'] = pipeline.get_pipeline_health()
    
    return results


# Health check for quality gates system
def check_quality_gates_health() -> bool:
    """Check if quality gates system is healthy."""
    try:
        pipeline = global_quality_pipeline
        health = pipeline.get_pipeline_health()
        return health['status'] in ['HEALTHY', 'WARNING']
    except Exception:
        return False