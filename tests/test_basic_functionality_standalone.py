"""Standalone tests that don't require external dependencies."""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports_without_dependencies():
    """Test that modules can be imported without external dependencies."""
    
    # Test core structure imports
    try:
        from pno_physics_bench.advanced_models import AdvancedPNORegistry
        from pno_physics_bench.autonomous_research_agent import ResearchHypothesis, HypothesisGenerator
        from pno_physics_bench.robustness.fault_tolerance import FaultReport, CircuitBreaker
        from pno_physics_bench.security_validation import SecurityThreat, InputValidator
        from pno_physics_bench.scaling.distributed_computing import ComputeNode, DistributedTask
        print("✓ All core modules imported successfully without dependencies")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    return True


def test_data_structures():
    """Test basic data structures work without external dependencies."""
    
    from pno_physics_bench.autonomous_research_agent import ResearchHypothesis
    
    # Test hypothesis creation
    hypothesis = ResearchHypothesis(
        id="test_001",
        title="Test Hypothesis",
        description="A test hypothesis",
        expected_improvement=0.1,
        complexity_score=5,
        priority=0.8,
        dependencies=[]
    )
    
    assert hypothesis.id == "test_001"
    assert hypothesis.status == "proposed"
    assert isinstance(hypothesis.evidence, dict)
    print("✓ ResearchHypothesis data structure works")
    
    from pno_physics_bench.robustness.fault_tolerance import FaultReport
    
    # Test fault report creation
    report = FaultReport(
        timestamp="2025-01-01",
        fault_type="test_fault",
        severity="medium",
        component="test_component",
        error_message="Test error",
        stack_trace="Test stack",
        recovery_action="test_recovery",
        success=False,
        performance_impact={"time": 1.0}
    )
    
    assert report.fault_type == "test_fault"
    assert report.success is False
    print("✓ FaultReport data structure works")
    
    from pno_physics_bench.security_validation import SecurityThreat
    
    # Test security threat creation
    threat = SecurityThreat(
        threat_id="threat_001",
        threat_type="malicious_input",
        severity="high",
        description="Test threat",
        source="test",
        timestamp=time.time(),
        mitigation="block",
        blocked=True
    )
    
    assert threat.threat_id == "threat_001"
    assert threat.blocked is True
    print("✓ SecurityThreat data structure works")
    
    return True


def test_algorithm_logic():
    """Test algorithm logic that doesn't require external libraries."""
    
    from pno_physics_bench.autonomous_research_agent import HypothesisGenerator
    
    generator = HypothesisGenerator()
    
    # Test hypothesis generation logic
    current_performance = {
        "accuracy": 0.8,
        "uncertainty": 0.7,
        "efficiency": 0.6
    }
    
    hypotheses = generator.generate_novel_hypotheses(current_performance, num_hypotheses=3)
    
    assert len(hypotheses) == 3
    assert all(h.expected_improvement > 0 for h in hypotheses)
    assert all(h.complexity_score > 0 for h in hypotheses)
    assert all(h.priority > 0 for h in hypotheses)
    print("✓ Hypothesis generation algorithm works")
    
    from pno_physics_bench.scaling.distributed_computing import LoadBalancer, ComputeNode, DistributedTask
    
    # Test load balancing logic
    load_balancer = LoadBalancer(balancing_strategy="round_robin")
    
    # Add nodes
    nodes = [
        ComputeNode(f"node_{i}", f"host{i}", 8000+i, {"cpu": 4}, status="idle")
        for i in range(3)
    ]
    
    for node in nodes:
        load_balancer.register_node(node)
    
    # Test node selection
    task = DistributedTask("test_task", "test_type", {})
    
    # Test round-robin selection
    selected_nodes = []
    for i in range(6):
        selected = load_balancer.select_node(task)
        selected_nodes.append(selected.node_id)
    
    # Should cycle through nodes
    assert selected_nodes[0] == selected_nodes[3]
    assert selected_nodes[1] == selected_nodes[4]  
    assert selected_nodes[2] == selected_nodes[5]
    print("✓ Load balancing algorithm works")
    
    return True


def test_fallback_implementations():
    """Test that fallback implementations work without PyTorch/NumPy."""
    
    from pno_physics_bench.advanced_models import AdvancedPNORegistry
    
    # Test registry functionality
    models = AdvancedPNORegistry.list_models()
    
    assert isinstance(models, dict)
    assert len(models) > 0
    print("✓ Model registry works without PyTorch")
    
    from pno_physics_bench.quantum_enhanced_pno import create_quantum_pno_suite
    
    # Test quantum suite creation (should work with fallbacks)
    config = {
        'input_channels': 3,
        'hidden_channels': 16,
        'quantum_config': {'num_qubits': 4}
    }
    
    suite = create_quantum_pno_suite(config)
    
    assert isinstance(suite, dict)
    assert 'tensor_network_pno' in suite  # Should always be available
    print("✓ Quantum PNO suite works with fallbacks")
    
    return True


def test_configuration_and_validation():
    """Test configuration handling and validation logic."""
    
    from pno_physics_bench.security_validation import InputValidator
    
    # Test validator configuration
    validator = InputValidator(
        max_tensor_size=1000,
        max_batch_size=10,
        allowed_dtypes=['float32', 'int32']
    )
    
    assert validator.max_tensor_size == 1000
    assert validator.max_batch_size == 10
    assert 'float32' in validator.allowed_dtypes
    print("✓ Input validator configuration works")
    
    # Test validation of basic Python types
    test_list = [1, 2, 3, 4, 5]
    is_valid, threat = validator.validate_tensor_input(test_list, "test_list")
    
    assert is_valid is True
    assert threat is None
    print("✓ Basic input validation works")
    
    # Test oversized input
    large_list = list(range(20))  # Larger than max_batch_size
    is_valid, threat = validator.validate_tensor_input(large_list, "large_list")
    
    assert is_valid is False
    assert threat is not None
    assert threat.threat_type == "excessive_sequence_length"
    print("✓ Size limit validation works")
    
    return True


def test_error_handling_and_recovery():
    """Test error handling without external dependencies."""
    
    from pno_physics_bench.robustness.fault_tolerance import CircuitBreaker, RetryStrategy
    
    # Test circuit breaker
    circuit_breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=0.1,
        expected_exception=RuntimeError
    )
    
    @circuit_breaker
    def failing_function():
        raise RuntimeError("Test failure")
    
    # Trip the circuit breaker
    failure_count = 0
    for i in range(3):
        try:
            failing_function()
        except RuntimeError:
            failure_count += 1
        except Exception as e:
            # Circuit breaker exception
            assert "Circuit breaker is OPEN" in str(e)
            break
    
    assert failure_count == 2  # Should trip after 2 failures
    assert circuit_breaker.state == "OPEN"
    print("✓ Circuit breaker logic works")
    
    # Test retry strategy
    retry = RetryStrategy(max_attempts=3, base_delay=0.01, jitter=False)
    
    call_count = 0
    
    @retry
    def eventually_succeeding_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError(f"Attempt {call_count}")
        return "Success"
    
    result = eventually_succeeding_function()
    assert result == "Success"
    assert call_count == 3
    print("✓ Retry strategy logic works")
    
    return True


def test_performance_and_caching_logic():
    """Test performance optimization logic without dependencies."""
    
    from pno_physics_bench.performance_optimization import ComputeCache
    
    # Test cache functionality
    cache = ComputeCache(max_entries=5, ttl_seconds=1.0)
    
    # Test basic operations
    assert cache.get("nonexistent") is None
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    
    # Test TTL
    cache.put("expire_test", "value", )
    time.sleep(1.1)  # Wait for expiration
    assert cache.get("expire_test") is None
    
    # Test eviction
    for i in range(6):  # More than max_entries
        cache.put(f"key{i}", f"value{i}")
    
    assert len(cache.cache) <= 5  # Should not exceed max_entries
    
    stats = cache.get_stats()
    assert isinstance(stats, dict)
    assert "hit_rate" in stats
    print("✓ Compute cache logic works")
    
    return True


def test_logging_and_monitoring():
    """Test logging functionality without dependencies."""
    
    from pno_physics_bench.comprehensive_logging import ExperimentMetric, ModelEvent
    
    # Test metric creation
    metric = ExperimentMetric(
        name="test_metric",
        value=0.95,
        step=10,
        timestamp=time.time(),
        experiment_id="test_exp",
        tags={"phase": "validation"}
    )
    
    assert metric.name == "test_metric"
    assert metric.value == 0.95
    assert metric.tags["phase"] == "validation"
    print("✓ Experiment metric structure works")
    
    # Test event creation
    event = ModelEvent(
        event_type="training_start",
        timestamp=time.time(),
        experiment_id="test_exp",
        details={"learning_rate": 0.001},
        severity="info"
    )
    
    assert event.event_type == "training_start"
    assert event.details["learning_rate"] == 0.001
    print("✓ Model event structure works")
    
    return True


def run_all_tests():
    """Run all standalone tests."""
    
    tests = [
        ("Module Imports", test_imports_without_dependencies),
        ("Data Structures", test_data_structures),
        ("Algorithm Logic", test_algorithm_logic),
        ("Fallback Implementations", test_fallback_implementations),
        ("Configuration and Validation", test_configuration_and_validation),
        ("Error Handling and Recovery", test_error_handling_and_recovery),
        ("Performance and Caching Logic", test_performance_and_caching_logic),
        ("Logging and Monitoring", test_logging_and_monitoring),
    ]
    
    results = []
    
    print("Running Standalone Tests (No External Dependencies)")
    print("=" * 60)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✓ {test_name}: PASSED")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name}: FAILED - {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    
    passed = sum(1 for name, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All standalone tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)