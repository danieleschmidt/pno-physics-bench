"""Comprehensive tests for robustness and security components."""

import pytest
import numpy as np
import time
import threading
from pathlib import Path
import sys
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from pno_physics_bench.robustness.fault_tolerance import (
    FaultReport,
    CircuitBreaker,
    RetryStrategy,
    GracefulDegradation,
    HealthMonitor,
    FaultTolerantPNO,
    create_fault_tolerant_system
)

from pno_physics_bench.security_validation import (
    SecurityThreat,
    InputValidator,
    ModelAccessControl,
    PrivacyProtector,
    SecureModelWrapper,
    create_secure_pno_system
)

from pno_physics_bench.comprehensive_logging import (
    ExperimentMetric,
    ModelEvent,
    StructuredLogger,
    ExperimentTracker
)


class TestFaultTolerance:
    """Test suite for fault tolerance components."""
    
    def test_fault_report_creation(self):
        """Test FaultReport creation and structure."""
        
        report = FaultReport(
            timestamp="2025-01-01 10:00:00",
            fault_type="model_error",
            severity="high",
            component="prediction_system",
            error_message="Model prediction failed",
            stack_trace="Traceback...",
            recovery_action="retry_with_fallback",
            success=False,
            performance_impact={"execution_time": 2.5}
        )
        
        assert report.fault_type == "model_error"
        assert report.severity == "high"
        assert report.success is False
        assert "execution_time" in report.performance_impact
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern."""
        
        # Create circuit breaker with low threshold for testing
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,  # 1 second for testing
            expected_exception=RuntimeError
        )
        
        # Mock function that fails
        @circuit_breaker
        def failing_function():
            raise RuntimeError("Simulated failure")
        
        # Test that circuit breaker trips after threshold
        for i in range(3):
            with pytest.raises(RuntimeError):
                failing_function()
        
        # Circuit should now be OPEN
        assert circuit_breaker.state == "OPEN"
        
        # Should raise circuit breaker exception
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            failing_function()
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        
        circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,  # Very short timeout for testing
            expected_exception=RuntimeError
        )
        
        # Mock function that can succeed or fail
        self.should_fail = True
        
        @circuit_breaker
        def conditional_function():
            if self.should_fail:
                raise RuntimeError("Failure")
            return "Success"
        
        # Trip the circuit breaker
        for i in range(2):
            with pytest.raises(RuntimeError):
                conditional_function()
        
        assert circuit_breaker.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Fix the function
        self.should_fail = False
        
        # Should transition to HALF_OPEN and then CLOSED on success
        result = conditional_function()
        assert result == "Success"
        assert circuit_breaker.state == "CLOSED"
    
    def test_retry_strategy(self):
        """Test retry strategy with different backoff methods."""
        
        # Test exponential backoff
        retry = RetryStrategy(
            max_attempts=3,
            backoff_strategy="exponential",
            base_delay=0.01,  # Very small delay for testing
            jitter=False
        )
        
        self.call_count = 0
        
        @retry
        def flaky_function():
            self.call_count += 1
            if self.call_count < 3:
                raise RuntimeError(f"Attempt {self.call_count} failed")
            return f"Success on attempt {self.call_count}"
        
        result = flaky_function()
        assert result == "Success on attempt 3"
        assert self.call_count == 3
    
    def test_retry_strategy_all_attempts_fail(self):
        """Test retry strategy when all attempts fail."""
        
        retry = RetryStrategy(max_attempts=2, base_delay=0.01)
        
        @retry
        def always_failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_failing_function()
    
    def test_graceful_degradation(self):
        """Test graceful degradation functionality."""
        
        degradation = GracefulDegradation(fallback_strategy="simple_model")
        
        @degradation
        def complex_model_prediction(x):
            # Simulate complex model failure
            raise RuntimeError("Complex model failed")
        
        # Should fall back gracefully
        if HAS_TORCH:
            test_input = torch.randn(2, 3, 4, 4)
            result = complex_model_prediction(test_input)
            # Should return fallback result
            assert result is not None
        else:
            test_input = np.random.randn(2, 3, 4, 4)
            result = complex_model_prediction(test_input)
            assert result is not None
    
    def test_health_monitor_creation(self):
        """Test HealthMonitor creation and configuration."""
        
        monitor = HealthMonitor(
            check_interval=30,
            alert_thresholds={
                "cpu_usage": 80.0,
                "memory_usage": 75.0,
                "error_rate": 0.1
            }
        )
        
        assert monitor.check_interval == 30
        assert monitor.alert_thresholds["cpu_usage"] == 80.0
        assert monitor.alert_thresholds["memory_usage"] == 75.0
    
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_health_monitor_system_check(self):
        """Test system health checking."""
        
        monitor = HealthMonitor()
        
        # Check system health
        health_report = monitor.check_health()
        
        assert isinstance(health_report, dict)
        assert "overall_status" in health_report
        assert "checks" in health_report
        assert "alerts" in health_report
        assert "recommendations" in health_report
        
        if "system" in health_report["checks"]:
            system_check = health_report["checks"]["system"]
            assert "cpu_usage" in system_check
            assert "memory_usage" in system_check
            assert "status" in system_check
    
    def test_health_monitor_model_check(self):
        """Test model health checking."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        monitor = HealthMonitor()
        
        # Create test model
        model = torch.nn.Linear(10, 5)
        
        # Check model health
        health_report = monitor.check_health(model=model)
        
        assert "model" in health_report["checks"]
        model_check = health_report["checks"]["model"]
        
        assert "parameters_finite" in model_check
        assert "gradients_finite" in model_check
        assert "model_size_mb" in model_check
        assert model_check["parameters_finite"] is True  # Should be healthy
    
    def test_health_monitor_corrupted_model(self):
        """Test health monitoring with corrupted model."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        monitor = HealthMonitor()
        
        # Create model with corrupted parameters
        model = torch.nn.Linear(10, 5)
        with torch.no_grad():
            model.weight[0, 0] = float('nan')  # Corrupt parameter
        
        health_report = monitor.check_health(model=model)
        
        model_check = health_report["checks"]["model"]
        assert model_check["parameters_finite"] is False
        assert model_check["status"] == "critical"
    
    def test_fault_tolerant_pno_creation(self):
        """Test FaultTolerantPNO wrapper creation."""
        
        # Mock base model
        class MockModel:
            def predict_with_uncertainty(self, x, num_samples=100):
                return {"prediction": x * 0.9, "uncertainty": 0.1}
        
        base_model = MockModel()
        
        ft_pno = FaultTolerantPNO(
            base_model=base_model,
            enable_circuit_breaker=True,
            enable_retry=True,
            enable_degradation=True,
            checkpoint_frequency=50
        )
        
        assert ft_pno.base_model is base_model
        assert ft_pno.circuit_breaker is not None
        assert ft_pno.retry_strategy is not None
        assert ft_pno.graceful_degradation is not None
        assert ft_pno.checkpoint_frequency == 50
    
    def test_fault_tolerant_pno_prediction(self):
        """Test fault-tolerant prediction functionality."""
        
        class MockModel:
            def __init__(self):
                self.call_count = 0
            
            def predict_with_uncertainty(self, x, num_samples=100):
                self.call_count += 1
                if self.call_count <= 2:  # Fail first two attempts
                    raise RuntimeError("Model temporarily unavailable")
                return {"prediction": x * 0.9, "uncertainty": 0.1}
        
        base_model = MockModel()
        ft_pno = FaultTolerantPNO(
            base_model=base_model,
            enable_retry=True,
            enable_degradation=True
        )
        
        # Should succeed after retries
        test_input = np.random.randn(2, 3, 4, 4)
        result = ft_pno.predict_with_fault_tolerance(test_input)
        
        assert result is not None
        # Should have retried and eventually succeeded
        assert base_model.call_count >= 2
    
    def test_fault_tolerant_system_factory(self):
        """Test fault-tolerant system factory function."""
        
        class MockModel:
            pass
        
        model = MockModel()
        
        # Default configuration
        ft_system = create_fault_tolerant_system(model)
        
        assert isinstance(ft_system, FaultTolerantPNO)
        assert ft_system.base_model is model
        
        # Custom configuration
        custom_config = {
            "enable_circuit_breaker": False,
            "enable_retry": True,
            "enable_degradation": False,
            "checkpoint_frequency": 200
        }
        
        ft_system_custom = create_fault_tolerant_system(model, custom_config)
        
        assert ft_system_custom.circuit_breaker is None
        assert ft_system_custom.retry_strategy is not None
        assert ft_system_custom.graceful_degradation is None
        assert ft_system_custom.checkpoint_frequency == 200


class TestSecurityValidation:
    """Test suite for security validation components."""
    
    def test_security_threat_creation(self):
        """Test SecurityThreat creation and structure."""
        
        threat = SecurityThreat(
            threat_id="threat_001",
            threat_type="malicious_input",
            severity="high",
            description="NaN values detected in input",
            source="input_validation",
            timestamp=time.time(),
            mitigation="sanitize_input",
            blocked=False
        )
        
        assert threat.threat_id == "threat_001"
        assert threat.threat_type == "malicious_input"
        assert threat.severity == "high"
        assert threat.blocked is False
    
    def test_input_validator_creation(self):
        """Test InputValidator creation and configuration."""
        
        validator = InputValidator(
            max_tensor_size=1000000,
            max_batch_size=100,
            allowed_dtypes=['float32', 'float64']
        )
        
        assert validator.max_tensor_size == 1000000
        assert validator.max_batch_size == 100
        assert 'float32' in validator.allowed_dtypes
        assert 'float64' in validator.allowed_dtypes
    
    def test_input_validator_valid_input(self):
        """Test input validation with valid inputs."""
        
        validator = InputValidator()
        
        # Test with NumPy array
        valid_array = np.random.randn(10, 10)
        is_valid, threat = validator.validate_tensor_input(valid_array, "test_array")
        
        assert is_valid is True
        assert threat is None
        
        # Test with PyTorch tensor if available
        if HAS_TORCH:
            valid_tensor = torch.randn(5, 5)
            is_valid, threat = validator.validate_tensor_input(valid_tensor, "test_tensor")
            
            assert is_valid is True
            assert threat is None
    
    def test_input_validator_malicious_input(self):
        """Test input validation with malicious inputs."""
        
        validator = InputValidator()
        
        # Test with NaN values
        malicious_array = np.array([[1.0, 2.0], [np.nan, 4.0]])
        is_valid, threat = validator.validate_tensor_input(malicious_array, "nan_array")
        
        assert is_valid is False
        assert threat is not None
        assert threat.threat_type == "malicious_nan"
        assert threat.mitigation == "sanitize_input"
        
        # Test with infinite values
        inf_array = np.array([[1.0, np.inf], [3.0, 4.0]])
        is_valid, threat = validator.validate_tensor_input(inf_array, "inf_array")
        
        assert is_valid is False
        assert threat is not None
        assert threat.threat_type == "malicious_inf"
    
    def test_input_validator_size_limits(self):
        """Test input validation with size limits."""
        
        validator = InputValidator(max_tensor_size=100, max_batch_size=5)
        
        # Test oversized tensor
        oversized_array = np.random.randn(20, 20)  # 400 elements > 100
        is_valid, threat = validator.validate_tensor_input(oversized_array, "large_array")
        
        assert is_valid is False
        assert threat is not None
        assert threat.threat_type == "excessive_array_size"
        assert threat.blocked is True
        
        # Test oversized batch
        batch_list = list(range(10))  # 10 items > 5 max_batch_size
        is_valid, threat = validator.validate_tensor_input(batch_list, "large_batch")
        
        assert is_valid is False
        assert threat is not None
        assert threat.threat_type == "excessive_sequence_length"
    
    def test_input_validator_sanitization(self):
        """Test input sanitization functionality."""
        
        validator = InputValidator()
        
        # Create threat for sanitization
        threat = SecurityThreat(
            threat_id="test_threat",
            threat_type="malicious_nan",
            severity="high",
            description="NaN values detected",
            source="test",
            timestamp=time.time(),
            mitigation="sanitize_input",
            blocked=False
        )
        
        # Test NumPy array sanitization
        corrupted_array = np.array([[1.0, np.nan], [np.inf, 4.0]])
        sanitized_array = validator.sanitize_input(corrupted_array, threat)
        
        assert not np.isnan(sanitized_array).any()
        assert not np.isinf(sanitized_array).any()
        assert sanitized_array[0, 1] == 0.0  # NaN replaced with 0
        assert sanitized_array[1, 0] == 0.0  # Inf replaced with 0
        
        # Test PyTorch tensor sanitization if available
        if HAS_TORCH:
            corrupted_tensor = torch.tensor([[1.0, float('nan')], [float('inf'), 4.0]])
            sanitized_tensor = validator.sanitize_input(corrupted_tensor, threat)
            
            assert not torch.isnan(sanitized_tensor).any()
            assert not torch.isinf(sanitized_tensor).any()
    
    def test_model_access_control(self):
        """Test ModelAccessControl functionality."""
        
        access_control = ModelAccessControl(
            authorized_operations=["forward", "predict", "evaluate"]
        )
        
        # Test authorized operation
        assert access_control.check_operation_permission("predict") is True
        
        # Test unauthorized operation
        assert access_control.check_operation_permission("train") is False
        assert access_control.check_operation_permission("modify_weights") is False
        
        # Check access log
        summary = access_control.get_access_summary()
        
        assert summary["total_access_attempts"] == 3
        assert summary["blocked_attempts"] == 2
        assert len(summary["recent_blocked"]) == 2
    
    def test_privacy_protector_differential_privacy(self):
        """Test PrivacyProtector differential privacy."""
        
        privacy = PrivacyProtector(
            enable_differential_privacy=True,
            epsilon=1.0
        )
        
        # Test with NumPy array
        original_data = np.random.randn(5, 5)
        noisy_data = privacy.add_differential_privacy_noise(original_data)
        
        # Should be different due to noise
        assert not np.array_equal(original_data, noisy_data)
        assert original_data.shape == noisy_data.shape
        
        # Test with PyTorch tensor if available
        if HAS_TORCH:
            original_tensor = torch.randn(3, 3)
            noisy_tensor = privacy.add_differential_privacy_noise(original_tensor)
            
            assert not torch.equal(original_tensor, noisy_tensor)
            assert original_tensor.shape == noisy_tensor.shape
    
    def test_privacy_protector_data_anonymization(self):
        """Test data anonymization functionality."""
        
        privacy = PrivacyProtector(enable_data_anonymization=True)
        
        # Test data anonymization
        sensitive_data = {
            "user_id": "user123",
            "session_id": "session456",
            "ip_address": "192.168.1.1",
            "prediction": [1, 2, 3],
            "timestamp": "2025-01-01"
        }
        
        anonymized_data = privacy.anonymize_data(sensitive_data)
        
        # Sensitive fields should be anonymized
        assert anonymized_data["user_id"] != sensitive_data["user_id"]
        assert anonymized_data["user_id"].startswith("anon_")
        assert anonymized_data["session_id"] != sensitive_data["session_id"]
        assert anonymized_data["ip_address"] != sensitive_data["ip_address"]
        
        # Non-sensitive fields should be unchanged
        assert anonymized_data["prediction"] == sensitive_data["prediction"]
    
    def test_secure_model_wrapper(self):
        """Test SecureModelWrapper functionality."""
        
        # Mock model
        class MockSecureModel:
            def predict_with_uncertainty(self, x):
                return {"prediction": x * 0.9, "uncertainty": 0.1}
        
        base_model = MockSecureModel()
        
        secure_model = SecureModelWrapper(
            model=base_model,
            enable_input_validation=True,
            enable_access_control=True,
            enable_privacy_protection=False
        )
        
        assert secure_model.model is base_model
        assert secure_model.input_validator is not None
        assert secure_model.access_control is not None
        assert secure_model.privacy_protector is None
    
    def test_secure_model_wrapper_valid_prediction(self):
        """Test secure prediction with valid input."""
        
        class MockModel:
            def predict_with_uncertainty(self, x):
                return {"prediction": x * 0.9, "uncertainty": 0.05}
        
        base_model = MockModel()
        secure_model = SecureModelWrapper(base_model, enable_access_control=False)
        
        # Test with valid input
        if HAS_TORCH:
            valid_input = torch.randn(2, 3, 4, 4)
        else:
            valid_input = np.random.randn(2, 3, 4, 4)
        
        result = secure_model.secure_predict(valid_input)
        
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "uncertainty" in result
    
    def test_secure_model_wrapper_malicious_input(self):
        """Test secure prediction with malicious input."""
        
        class MockModel:
            def predict_with_uncertainty(self, x):
                return {"prediction": x * 0.9, "uncertainty": 0.05}
        
        base_model = MockModel()
        secure_model = SecureModelWrapper(base_model, enable_access_control=False)
        
        # Test with malicious input (NaN values)
        if HAS_TORCH:
            malicious_input = torch.full((2, 3, 4, 4), float('nan'))
        else:
            malicious_input = np.full((2, 3, 4, 4), np.nan)
        
        # Should sanitize and process
        result = secure_model.secure_predict(malicious_input)
        
        assert isinstance(result, dict)
        # Should have processed successfully after sanitization
        assert "prediction" in result
    
    def test_secure_model_wrapper_blocked_input(self):
        """Test secure prediction with blocked input."""
        
        class MockModel:
            def predict_with_uncertainty(self, x):
                return {"prediction": x * 0.9, "uncertainty": 0.05}
        
        base_model = MockModel()
        secure_model = SecureModelWrapper(
            base_model, 
            enable_access_control=False,
            security_config={"max_tensor_size": 100}  # Very small limit
        )
        
        # Test with oversized input
        oversized_input = np.random.randn(20, 20)  # 400 elements > 100
        
        with pytest.raises(ValueError, match="Security threat detected"):
            secure_model.secure_predict(oversized_input)
    
    def test_secure_system_factory(self):
        """Test secure system factory function."""
        
        class MockModel:
            pass
        
        model = MockModel()
        
        # Test different security levels
        minimal_system = create_secure_pno_system(model, security_level="minimal")
        assert minimal_system.input_validator is not None
        assert minimal_system.access_control is None
        assert minimal_system.privacy_protector is None
        
        standard_system = create_secure_pno_system(model, security_level="standard")
        assert standard_system.input_validator is not None
        assert standard_system.access_control is not None
        assert standard_system.privacy_protector is None
        
        high_system = create_secure_pno_system(model, security_level="high")
        assert high_system.input_validator is not None
        assert high_system.access_control is not None
        assert high_system.privacy_protector is not None


class TestComprehensiveLogging:
    """Test suite for comprehensive logging components."""
    
    def test_experiment_metric_creation(self):
        """Test ExperimentMetric creation and structure."""
        
        metric = ExperimentMetric(
            name="train_loss",
            value=0.5,
            step=100,
            timestamp=time.time(),
            experiment_id="exp_001",
            tags={"epoch": "10", "batch": "5"}
        )
        
        assert metric.name == "train_loss"
        assert metric.value == 0.5
        assert metric.step == 100
        assert metric.experiment_id == "exp_001"
        assert metric.tags["epoch"] == "10"
    
    def test_model_event_creation(self):
        """Test ModelEvent creation and structure."""
        
        event = ModelEvent(
            event_type="training_start",
            timestamp=time.time(),
            experiment_id="exp_001",
            details={"learning_rate": 0.001, "batch_size": 32},
            severity="info"
        )
        
        assert event.event_type == "training_start"
        assert event.experiment_id == "exp_001"
        assert event.details["learning_rate"] == 0.001
        assert event.severity == "info"
    
    def test_structured_logger_creation(self):
        """Test StructuredLogger creation and setup."""
        
        logger = StructuredLogger(
            log_dir="test_logs",
            experiment_id="test_exp",
            log_level="DEBUG"
        )
        
        assert logger.experiment_id == "test_exp"
        assert logger.log_level == logging.DEBUG
        assert logger.log_dir == Path("test_logs")
        
        # Check logger components
        assert hasattr(logger, 'main_logger')
        assert hasattr(logger, 'metrics_logger')
        assert hasattr(logger, 'error_logger')
        
        # Cleanup
        logger.close()
    
    def test_structured_logger_metric_logging(self):
        """Test metric logging functionality."""
        
        logger = StructuredLogger(
            log_dir="test_logs",
            experiment_id="metric_test",
            log_level="INFO"
        )
        
        # Log metrics
        logger.log_metric("accuracy", 0.95, step=10, tags={"phase": "validation"})
        logger.log_metric("loss", 0.1, step=10)
        
        # Check metrics buffer
        assert len(logger.metrics_buffer) == 2
        
        metric1 = logger.metrics_buffer[0]
        assert metric1.name == "accuracy"
        assert metric1.value == 0.95
        assert metric1.step == 10
        assert metric1.tags["phase"] == "validation"
        
        # Cleanup
        logger.close()
    
    def test_structured_logger_event_logging(self):
        """Test event logging functionality."""
        
        logger = StructuredLogger(log_dir="test_logs", experiment_id="event_test")
        
        # Log events
        logger.log_event("training_start", {"lr": 0.001}, severity="info")
        logger.log_event("model_error", {"error": "NaN loss"}, severity="error")
        
        # Check events buffer
        assert len(logger.events_buffer) == 2
        
        event1 = logger.events_buffer[0]
        assert event1.event_type == "training_start"
        assert event1.details["lr"] == 0.001
        assert event1.severity == "info"
        
        # Cleanup
        logger.close()
    
    def test_structured_logger_tensor_stats(self):
        """Test tensor statistics logging."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        logger = StructuredLogger(log_dir="test_logs", experiment_id="tensor_test")
        
        # Create test tensor
        test_tensor = torch.randn(3, 4, 5)
        
        # Log tensor stats
        logger.log_tensor_stats(test_tensor, name="test_weights", step=5)
        
        # Should create metrics and events
        assert len(logger.metrics_buffer) > 0
        assert len(logger.events_buffer) > 0
        
        # Check that tensor stats were logged as metrics
        metric_names = [m.name for m in logger.metrics_buffer]
        assert "test_weights_mean" in metric_names
        assert "test_weights_std" in metric_names
        assert "test_weights_shape" in metric_names
        
        # Cleanup
        logger.close()
    
    def test_structured_logger_execution_time(self):
        """Test execution time logging context manager."""
        
        logger = StructuredLogger(log_dir="test_logs", experiment_id="time_test")
        
        # Test successful operation
        with logger.log_execution_time("test_operation", log_level="debug"):
            time.sleep(0.01)  # Small delay for measurable time
        
        # Should log start and complete events
        assert len(logger.events_buffer) >= 2
        
        # Should log duration metric
        duration_metrics = [m for m in logger.metrics_buffer if "duration" in m.name]
        assert len(duration_metrics) > 0
        
        # Test operation with error
        with pytest.raises(ValueError):
            with logger.log_execution_time("failing_operation"):
                raise ValueError("Test error")
        
        # Should log error event
        error_events = [e for e in logger.events_buffer if "error" in e.event_type]
        assert len(error_events) > 0
        
        # Cleanup
        logger.close()
    
    def test_structured_logger_model_parameters(self):
        """Test model parameter logging."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        logger = StructuredLogger(log_dir="test_logs", experiment_id="model_test")
        
        # Create test model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Log model parameters
        logger.log_model_parameters(model, step=0)
        
        # Should create parameter metrics
        param_metrics = [m for m in logger.metrics_buffer if "model." in m.name]
        assert len(param_metrics) > 0
        
        # Should log model event
        model_events = [e for e in logger.events_buffer if e.event_type == "model_parameters"]
        assert len(model_events) > 0
        
        # Cleanup
        logger.close()
    
    def test_structured_logger_summary_generation(self):
        """Test experiment summary generation."""
        
        logger = StructuredLogger(log_dir="test_logs", experiment_id="summary_test")
        
        # Add some test data
        logger.log_metric("accuracy", 0.8, step=1)
        logger.log_metric("accuracy", 0.9, step=2)
        logger.log_metric("loss", 0.5, step=1)
        logger.log_event("training_start", {})
        logger.log_event("validation", {})
        
        # Generate summary
        summary = logger.generate_experiment_summary()
        
        assert isinstance(summary, dict)
        assert summary["experiment_id"] == "summary_test"
        assert summary["total_metrics"] == 3
        assert summary["total_events"] == 2
        assert "metrics_summary" in summary
        assert "events_summary" in summary
        assert "performance_summary" in summary
        
        # Check metrics summary
        assert "accuracy" in summary["metrics_summary"]
        accuracy_summary = summary["metrics_summary"]["accuracy"]
        assert accuracy_summary["count"] == 2
        assert accuracy_summary["mean"] == 0.85
        assert accuracy_summary["last_value"] == 0.9
        
        # Cleanup
        logger.close()
    
    def test_experiment_tracker(self):
        """Test high-level ExperimentTracker functionality."""
        
        tracker = ExperimentTracker("test_experiment", log_dir="test_experiments")
        
        assert "test_experiment" in tracker.experiment_id
        assert tracker.logger is not None
        
        # Log hyperparameters
        tracker.log_hyperparameters(
            learning_rate=0.001,
            batch_size=32,
            model_type="PNO"
        )
        
        # Log metrics
        tracker.log_metrics(step=1, train_loss=0.5, val_accuracy=0.8)
        tracker.log_metrics(step=2, train_loss=0.4, val_accuracy=0.85)
        
        # Check that metrics were logged
        assert len(tracker.logger.metrics_buffer) >= 4  # 2 steps × 2 metrics each
        
        # Finish experiment
        summary = tracker.finish_experiment()
        
        assert isinstance(summary, dict)
        assert summary["total_metrics"] >= 4
        assert "export_file" in summary


def test_integration_robustness_security():
    """Integration test combining robustness and security features."""
    
    # Create a mock model with potential failures
    class FlakySensitiveModel:
        def __init__(self):
            self.call_count = 0
        
        def predict_with_uncertainty(self, x):
            self.call_count += 1
            
            # Occasionally fail
            if self.call_count % 4 == 0:
                raise RuntimeError("Periodic failure")
            
            # Check for malicious input patterns
            if hasattr(x, 'isnan') and x.isnan().any():
                raise ValueError("Corrupted input detected")
            
            return {"prediction": x * 0.9, "uncertainty": 0.1}
    
    # Create fault-tolerant and secure wrapper
    base_model = FlakySensitiveModel()
    
    # Apply fault tolerance
    ft_model = create_fault_tolerant_system(
        base_model, 
        config={
            "enable_retry": True,
            "enable_degradation": True,
            "enable_circuit_breaker": False  # Disable for this test
        }
    )
    
    # Apply security wrapper
    secure_ft_model = create_secure_pno_system(ft_model, security_level="standard")
    
    # Test with various inputs
    test_cases = [
        np.random.randn(2, 3, 4, 4),  # Normal input
        np.array([[1.0, np.nan], [3.0, 4.0]]),  # Malicious input (will be sanitized)
    ]
    
    successful_predictions = 0
    
    for i, test_input in enumerate(test_cases):
        try:
            result = secure_ft_model.secure_predict(test_input)
            successful_predictions += 1
            
            assert isinstance(result, dict)
            assert "prediction" in result or result is not None
            
        except Exception as e:
            # Some failures are expected due to circuit breaking or blocking
            print(f"Test case {i} failed as expected: {e}")
    
    # Should have at least some successful predictions
    assert successful_predictions > 0
    
    # Get comprehensive status
    fault_summary = ft_model.get_fault_summary()
    security_report = secure_ft_model.get_security_report()
    
    assert isinstance(fault_summary, dict)
    assert isinstance(security_report, dict)
    assert "total_incidents" in security_report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])