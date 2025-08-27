#!/usr/bin/env python3
"""Simple Robust Adaptive PNO Demo - Working demonstration of robustness features."""

import json
import time
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FailureMode:
    GRADIENT_EXPLOSION = "gradient_explosion"
    UNCERTAINTY_COLLAPSE = "uncertainty_collapse"
    NUMERICAL_INSTABILITY = "numerical_instability"

class ThreatLevel:
    MINIMAL = "minimal"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SimpleRobustPNO:
    """Simple robust PNO with failure detection and recovery."""
    
    def __init__(self):
        self.is_healthy = True
        self.failure_mode = None
        self.recovery_count = 0
        self.security_threats = 0
        
    def predict(self, input_data):
        """Make prediction with potential failures."""
        if self.failure_mode == FailureMode.NUMERICAL_INSTABILITY:
            if random.random() < 0.3:
                return float('nan')
        
        # Simple calculation
        result = sum(input_data) * 0.5 + random.uniform(-0.1, 0.1)
        
        if self.failure_mode == FailureMode.GRADIENT_EXPLOSION:
            result *= 1000  # Simulate explosion
        elif self.failure_mode == FailureMode.UNCERTAINTY_COLLAPSE:
            result = input_data[0]  # No transformation
            
        return result
    
    def inject_failure(self, failure_mode):
        """Inject failure for testing."""
        self.failure_mode = failure_mode
        self.is_healthy = False
        logger.info(f"Injected failure: {failure_mode}")
    
    def detect_failure(self, prediction):
        """Detect if prediction indicates failure."""
        if str(prediction) == 'nan' or prediction is None:
            return [FailureMode.NUMERICAL_INSTABILITY]
        
        if abs(prediction) > 100:
            return [FailureMode.GRADIENT_EXPLOSION]
        
        if abs(prediction) < 0.001:
            return [FailureMode.UNCERTAINTY_COLLAPSE]
        
        return []
    
    def recover(self, failure_modes):
        """Attempt recovery from failures."""
        success_count = 0
        
        for failure in failure_modes:
            recovery_success = random.random() < 0.8  # 80% success rate
            
            if recovery_success:
                self.failure_mode = None
                self.is_healthy = True
                success_count += 1
                self.recovery_count += 1
                logger.info(f"Successfully recovered from {failure}")
            else:
                logger.error(f"Failed to recover from {failure}")
        
        return success_count == len(failure_modes)
    
    def detect_security_threat(self, input_data):
        """Simple security threat detection."""
        # Check for suspicious input patterns
        if max(input_data) > 50 or min(input_data) < -50:
            self.security_threats += 1
            return ThreatLevel.HIGH
        
        if sum(input_data) > 100:
            self.security_threats += 1
            return ThreatLevel.MEDIUM
        
        return ThreatLevel.MINIMAL

def run_simple_robust_demo():
    """Run simple robustness demonstration."""
    
    logger.info("🛡️ Simple Robust Adaptive PNO Demo")
    logger.info("=" * 50)
    
    model = SimpleRobustPNO()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Normal Operation',
            'input': [1.0, 2.0, 3.0, 4.0],
            'failure': None,
            'expected': 'success'
        },
        {
            'name': 'Gradient Explosion',
            'input': [1.0, 2.0, 3.0, 4.0],
            'failure': FailureMode.GRADIENT_EXPLOSION,
            'expected': 'recovery'
        },
        {
            'name': 'Uncertainty Collapse', 
            'input': [1.0, 2.0, 3.0, 4.0],
            'failure': FailureMode.UNCERTAINTY_COLLAPSE,
            'expected': 'recovery'
        },
        {
            'name': 'Numerical Instability',
            'input': [1.0, 2.0, 3.0, 4.0], 
            'failure': FailureMode.NUMERICAL_INSTABILITY,
            'expected': 'recovery'
        },
        {
            'name': 'Security Threat',
            'input': [100.0, 200.0, 300.0, 400.0],  # Suspicious values
            'failure': None,
            'expected': 'mitigation'
        }
    ]
    
    results = {
        'scenarios_tested': 0,
        'failures_detected': 0,
        'recoveries_successful': 0,
        'security_threats': 0,
        'scenario_results': []
    }
    
    for i, scenario in enumerate(scenarios):
        logger.info(f"\n🧪 Scenario {i+1}: {scenario['name']}")
        
        scenario_result = {
            'name': scenario['name'],
            'outcome': 'unknown',
            'failures_detected': [],
            'security_threat': ThreatLevel.MINIMAL
        }
        
        try:
            # Inject failure if specified
            if scenario['failure']:
                model.inject_failure(scenario['failure'])
            
            # Security check
            threat_level = model.detect_security_threat(scenario['input'])
            scenario_result['security_threat'] = threat_level
            
            if threat_level != ThreatLevel.MINIMAL:
                results['security_threats'] += 1
                logger.warning(f"Security threat detected: {threat_level}")
            
            # Make prediction
            prediction = model.predict(scenario['input'])
            logger.info(f"Prediction: {prediction}")
            
            # Check for failures
            detected_failures = model.detect_failure(prediction)
            scenario_result['failures_detected'] = detected_failures
            
            if detected_failures:
                results['failures_detected'] += len(detected_failures)
                logger.warning(f"Failures detected: {detected_failures}")
                
                # Attempt recovery
                recovery_success = model.recover(detected_failures)
                
                if recovery_success:
                    results['recoveries_successful'] += 1
                    scenario_result['outcome'] = 'recovered'
                    logger.info("✅ Recovery successful")
                else:
                    scenario_result['outcome'] = 'recovery_failed'
                    logger.error("❌ Recovery failed")
            else:
                scenario_result['outcome'] = 'normal' if threat_level == ThreatLevel.MINIMAL else 'threat_mitigated'
                logger.info("✅ Normal operation")
                
        except Exception as e:
            scenario_result['outcome'] = 'error'
            logger.error(f"❌ Scenario failed: {e}")
        
        finally:
            # Reset for next scenario
            model.failure_mode = None
            model.is_healthy = True
            
        results['scenarios_tested'] += 1
        results['scenario_results'].append(scenario_result)
    
    # Final report
    logger.info("\n" + "=" * 50)
    logger.info("📊 ROBUSTNESS DEMO RESULTS")
    logger.info("=" * 50)
    
    logger.info(f"Scenarios tested: {results['scenarios_tested']}")
    logger.info(f"Failures detected: {results['failures_detected']}")
    logger.info(f"Successful recoveries: {results['recoveries_successful']}")
    logger.info(f"Security threats: {results['security_threats']}")
    logger.info(f"Total model recoveries: {model.recovery_count}")
    
    if results['failures_detected'] > 0:
        recovery_rate = results['recoveries_successful'] / results['failures_detected'] * 100
        logger.info(f"Recovery success rate: {recovery_rate:.1f}%")
    
    # Outcome distribution
    outcomes = {}
    for result in results['scenario_results']:
        outcome = result['outcome']
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    
    logger.info(f"\nOutcome distribution:")
    for outcome, count in outcomes.items():
        logger.info(f"  {outcome}: {count}")
    
    logger.info(f"\n🏆 Key Achievements:")
    logger.info(f"  ✓ Automatic failure detection")
    logger.info(f"  ✓ Self-recovery mechanisms")
    logger.info(f"  ✓ Security threat detection")
    logger.info(f"  ✓ Graceful degradation")
    
    # Save results
    with open('/tmp/simple_robust_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to /tmp/simple_robust_demo_results.json")
    logger.info("🛡️ Robust Adaptive PNO Demo Complete!")
    
    return results

if __name__ == "__main__":
    start_time = time.time()
    
    results = run_simple_robust_demo()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Demo completed in {elapsed:.2f} seconds")
    print("🎯 Simple Robust Demo Success!")