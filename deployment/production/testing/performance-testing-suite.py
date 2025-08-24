#!/usr/bin/env python3
"""
PNO Physics Bench - Performance Testing Suite
Comprehensive load testing, capacity planning, and SLA validation
"""

import asyncio
import aiohttp
import time
import json
import logging
import statistics
import subprocess
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Individual test result"""
    timestamp: float
    response_time: float
    status_code: int
    error: Optional[str] = None
    request_size: int = 0
    response_size: int = 0

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    url: str
    method: str = "GET"
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    payload: Optional[Dict] = None
    headers: Optional[Dict] = None
    timeout: int = 30

@dataclass
class TestSummary:
    """Test execution summary"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    errors: Dict[str, int]
    start_time: str
    end_time: str
    duration: float

class PerformanceTester:
    """Comprehensive performance testing framework"""
    
    def __init__(self, base_url: str = "https://api.pno-physics-bench.com"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        
    async def make_request(self, session: aiohttp.ClientSession, config: LoadTestConfig) -> TestResult:
        """Make a single HTTP request"""
        start_time = time.time()
        
        try:
            kwargs = {
                'timeout': aiohttp.ClientTimeout(total=config.timeout),
                'headers': config.headers or {}
            }
            
            if config.payload and config.method in ['POST', 'PUT', 'PATCH']:
                kwargs['json'] = config.payload
            
            async with session.request(config.method, config.url, **kwargs) as response:
                response_body = await response.read()
                end_time = time.time()
                
                return TestResult(
                    timestamp=start_time,
                    response_time=(end_time - start_time) * 1000,  # Convert to ms
                    status_code=response.status,
                    request_size=len(json.dumps(config.payload) if config.payload else 0),
                    response_size=len(response_body)
                )
                
        except Exception as e:
            end_time = time.time()
            return TestResult(
                timestamp=start_time,
                response_time=(end_time - start_time) * 1000,
                status_code=0,
                error=str(e)
            )
    
    async def run_load_test(self, config: LoadTestConfig) -> TestSummary:
        """Run load test with specified configuration"""
        logger.info(f"Starting load test: {config.concurrent_users} users for {config.duration_seconds}s")
        
        start_time = time.time()
        results = []
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(config.concurrent_users)
        
        async def bounded_request(session):
            async with semaphore:
                return await self.make_request(session, config)
        
        # Run test
        connector = aiohttp.TCPConnector(limit=config.concurrent_users * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            end_time = start_time + config.duration_seconds
            
            # Ramp up period
            ramp_up_delay = config.ramp_up_seconds / config.concurrent_users
            
            tasks = []
            while time.time() < end_time:
                # Add tasks up to concurrent user limit
                while len(tasks) < config.concurrent_users and time.time() < end_time:
                    task = asyncio.create_task(bounded_request(session))
                    tasks.append(task)
                    
                    # Ramp up delay
                    if len(tasks) <= config.concurrent_users:
                        await asyncio.sleep(ramp_up_delay)
                
                # Wait for some tasks to complete
                if tasks:
                    done, pending = await asyncio.wait(tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                    
                    for task in done:
                        result = await task
                        results.append(result)
                        tasks.remove(task)
            
            # Wait for remaining tasks
            if tasks:
                remaining_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in remaining_results:
                    if isinstance(result, TestResult):
                        results.append(result)
        
        # Calculate summary
        return self._calculate_summary(results, start_time, time.time())
    
    def _calculate_summary(self, results: List[TestResult], start_time: float, end_time: float) -> TestSummary:
        """Calculate test summary from results"""
        if not results:
            return TestSummary(
                total_requests=0, successful_requests=0, failed_requests=0,
                avg_response_time=0, min_response_time=0, max_response_time=0,
                p50_response_time=0, p95_response_time=0, p99_response_time=0,
                requests_per_second=0, errors={}, 
                start_time=datetime.fromtimestamp(start_time).isoformat(),
                end_time=datetime.fromtimestamp(end_time).isoformat(),
                duration=end_time - start_time
            )
        
        # Filter successful requests
        successful_results = [r for r in results if r.status_code == 200]
        failed_results = [r for r in results if r.status_code != 200 or r.error]
        
        # Response times for successful requests
        response_times = [r.response_time for r in successful_results]
        
        # Error analysis
        errors = {}
        for result in failed_results:
            if result.error:
                errors[result.error] = errors.get(result.error, 0) + 1
            else:
                status_key = f"HTTP_{result.status_code}"
                errors[status_key] = errors.get(status_key, 0) + 1
        
        # Calculate percentiles
        if response_times:
            p50 = np.percentile(response_times, 50)
            p95 = np.percentile(response_times, 95)
            p99 = np.percentile(response_times, 99)
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
        else:
            p50 = p95 = p99 = avg_time = min_time = max_time = 0
        
        duration = end_time - start_time
        rps = len(results) / duration if duration > 0 else 0
        
        return TestSummary(
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            avg_response_time=avg_time,
            min_response_time=min_time,
            max_response_time=max_time,
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            requests_per_second=rps,
            errors=errors,
            start_time=datetime.fromtimestamp(start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            duration=duration
        )

class CapacityPlanner:
    """Capacity planning and resource estimation"""
    
    def __init__(self, performance_tester: PerformanceTester):
        self.tester = performance_tester
        
    async def run_capacity_analysis(self, base_url: str) -> Dict[str, Any]:
        """Run comprehensive capacity analysis"""
        logger.info("Starting capacity planning analysis")
        
        capacity_results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_performance': await self._baseline_test(base_url),
            'load_progression': await self._load_progression_test(base_url),
            'stress_test': await self._stress_test(base_url),
            'endurance_test': await self._endurance_test(base_url),
            'recommendations': {}
        }
        
        # Generate capacity recommendations
        capacity_results['recommendations'] = self._generate_capacity_recommendations(capacity_results)
        
        return capacity_results
    
    async def _baseline_test(self, base_url: str) -> Dict[str, Any]:
        """Establish baseline performance metrics"""
        logger.info("Running baseline performance test")
        
        config = LoadTestConfig(
            url=f"{base_url}/health",
            concurrent_users=1,
            duration_seconds=60,
            ramp_up_seconds=0
        )
        
        summary = await self.tester.run_load_test(config)
        
        return {
            'test_type': 'baseline',
            'configuration': asdict(config),
            'results': asdict(summary)
        }
    
    async def _load_progression_test(self, base_url: str) -> List[Dict[str, Any]]:
        """Test performance under increasing load"""
        logger.info("Running load progression test")
        
        user_levels = [1, 5, 10, 20, 50, 100, 200]
        progression_results = []
        
        for users in user_levels:
            logger.info(f"Testing with {users} concurrent users")
            
            config = LoadTestConfig(
                url=f"{base_url}/predict",
                method="POST",
                concurrent_users=users,
                duration_seconds=120,
                ramp_up_seconds=10,
                payload={
                    "input_data": [[1.0, 2.0, 3.0]] * 100,  # Sample ML input
                    "model_config": {"uncertainty": True}
                }
            )
            
            summary = await self.tester.run_load_test(config)
            
            progression_results.append({
                'concurrent_users': users,
                'configuration': asdict(config),
                'results': asdict(summary),
                'performance_degradation': self._calculate_degradation(summary, users)
            })
            
            # Break if error rate becomes too high
            if summary.failed_requests / summary.total_requests > 0.05:  # 5% error rate
                logger.warning(f"High error rate at {users} users, stopping progression test")
                break
            
            # Brief pause between tests
            await asyncio.sleep(10)
        
        return progression_results
    
    async def _stress_test(self, base_url: str) -> Dict[str, Any]:
        """Find breaking point of the system"""
        logger.info("Running stress test to find system limits")
        
        # Start with aggressive load
        config = LoadTestConfig(
            url=f"{base_url}/predict",
            method="POST",
            concurrent_users=500,
            duration_seconds=300,
            ramp_up_seconds=30,
            payload={
                "input_data": [[1.0, 2.0, 3.0]] * 200,
                "model_config": {"uncertainty": True, "batch_size": 32}
            }
        )
        
        summary = await self.tester.run_load_test(config)
        
        return {
            'test_type': 'stress',
            'configuration': asdict(config),
            'results': asdict(summary),
            'system_limits': {
                'max_concurrent_users': config.concurrent_users,
                'max_rps': summary.requests_per_second,
                'failure_point_error_rate': summary.failed_requests / summary.total_requests if summary.total_requests > 0 else 0
            }
        }
    
    async def _endurance_test(self, base_url: str) -> Dict[str, Any]:
        """Test system stability over extended period"""
        logger.info("Running endurance test (30 minutes)")
        
        config = LoadTestConfig(
            url=f"{base_url}/predict",
            method="POST",
            concurrent_users=50,
            duration_seconds=1800,  # 30 minutes
            ramp_up_seconds=60,
            payload={
                "input_data": [[1.0, 2.0, 3.0]] * 50,
                "model_config": {"uncertainty": True}
            }
        )
        
        summary = await self.tester.run_load_test(config)
        
        return {
            'test_type': 'endurance',
            'configuration': asdict(config),
            'results': asdict(summary),
            'stability_metrics': {
                'performance_consistency': self._calculate_consistency(summary),
                'memory_leak_indicator': await self._check_memory_usage(),
                'error_rate_trend': 'stable' if summary.failed_requests / summary.total_requests < 0.01 else 'increasing'
            }
        }
    
    def _calculate_degradation(self, summary: TestSummary, concurrent_users: int) -> float:
        """Calculate performance degradation percentage"""
        # Baseline expectation: 100ms response time for 1 user
        baseline_expected = 100.0
        
        # Calculate expected response time based on user load (linear approximation)
        expected_response_time = baseline_expected * (1 + (concurrent_users - 1) * 0.1)
        
        if expected_response_time == 0:
            return 0
        
        degradation = (summary.avg_response_time - expected_response_time) / expected_response_time * 100
        return max(0, degradation)
    
    def _calculate_consistency(self, summary: TestSummary) -> str:
        """Calculate performance consistency rating"""
        if summary.avg_response_time == 0:
            return "no_data"
        
        # Calculate coefficient of variation (std dev / mean)
        cv = (summary.max_response_time - summary.min_response_time) / summary.avg_response_time
        
        if cv < 0.2:
            return "excellent"
        elif cv < 0.5:
            return "good"
        elif cv < 1.0:
            return "fair"
        else:
            return "poor"
    
    async def _check_memory_usage(self) -> str:
        """Check for potential memory leaks"""
        try:
            # Get memory usage from Kubernetes
            cmd = "kubectl top pods -n production -l app=pno-physics-bench --no-headers"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                memory_values = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split()
                        memory_mb = int(parts[2].replace('Mi', ''))
                        memory_values.append(memory_mb)
                
                avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0
                
                # Simple heuristic: if average memory usage > 80% of limit, potential leak
                if avg_memory > 3200:  # 80% of 4Gi
                    return "potential_leak"
                else:
                    return "stable"
            else:
                return "unable_to_check"
                
        except Exception as e:
            logger.warning(f"Unable to check memory usage: {e}")
            return "unable_to_check"
    
    def _generate_capacity_recommendations(self, capacity_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate capacity planning recommendations"""
        recommendations = {
            'scaling_recommendations': [],
            'resource_recommendations': [],
            'sla_compliance': {},
            'cost_optimization': []
        }
        
        # Analyze load progression results
        progression = capacity_results.get('load_progression', [])
        if progression:
            # Find optimal concurrent user level
            optimal_users = 1
            for result in progression:
                if (result['results']['p95_response_time'] < 500 and  # Under 500ms P95
                    result['results']['failed_requests'] / result['results']['total_requests'] < 0.01):  # Under 1% error
                    optimal_users = result['concurrent_users']
            
            recommendations['scaling_recommendations'].append({
                'metric': 'concurrent_users',
                'optimal_level': optimal_users,
                'max_tested': progression[-1]['concurrent_users'],
                'recommendation': f"System can handle {optimal_users} concurrent users while maintaining SLA"
            })
        
        # Resource recommendations
        stress_results = capacity_results.get('stress_test', {})
        if stress_results:
            max_rps = stress_results.get('system_limits', {}).get('max_rps', 0)
            
            recommendations['resource_recommendations'].append({
                'metric': 'requests_per_second',
                'current_capacity': max_rps,
                'recommended_headroom': max_rps * 0.3,  # 30% headroom
                'scaling_trigger': max_rps * 0.7,  # Scale at 70% capacity
                'recommendation': f"Configure auto-scaling to trigger at {max_rps * 0.7:.1f} RPS"
            })
        
        # SLA compliance analysis
        sla_targets = {
            'response_time_p95': 500,  # 500ms
            'availability': 99.9,      # 99.9%
            'error_rate': 0.1          # 0.1%
        }
        
        for test_type, results in capacity_results.items():
            if isinstance(results, dict) and 'results' in results:
                test_results = results['results']
                
                # Check P95 response time
                p95_compliant = test_results.get('p95_response_time', 0) <= sla_targets['response_time_p95']
                
                # Check error rate
                total_req = test_results.get('total_requests', 1)
                failed_req = test_results.get('failed_requests', 0)
                error_rate = (failed_req / total_req) * 100 if total_req > 0 else 0
                error_rate_compliant = error_rate <= sla_targets['error_rate']
                
                recommendations['sla_compliance'][test_type] = {
                    'p95_response_time': {
                        'target': sla_targets['response_time_p95'],
                        'actual': test_results.get('p95_response_time', 0),
                        'compliant': p95_compliant
                    },
                    'error_rate': {
                        'target': sla_targets['error_rate'],
                        'actual': error_rate,
                        'compliant': error_rate_compliant
                    }
                }
        
        return recommendations

class SLAValidator:
    """Service Level Agreement validation"""
    
    def __init__(self):
        self.sla_targets = {
            'availability': 99.9,           # 99.9% uptime
            'response_time_p95': 500,       # 500ms 95th percentile
            'response_time_p99': 1000,      # 1s 99th percentile
            'error_rate': 0.1,              # 0.1% error rate
            'throughput_min': 100           # 100 RPS minimum
        }
    
    def validate_sla_compliance(self, test_results: List[TestSummary]) -> Dict[str, Any]:
        """Validate SLA compliance across test results"""
        logger.info("Validating SLA compliance")
        
        compliance_results = {
            'timestamp': datetime.now().isoformat(),
            'sla_targets': self.sla_targets,
            'compliance_status': {},
            'overall_compliance': True,
            'violations': [],
            'recommendations': []
        }
        
        for test_result in test_results:
            test_compliance = self._validate_single_test(test_result)
            test_name = f"test_{test_result.start_time}"
            compliance_results['compliance_status'][test_name] = test_compliance
            
            if not test_compliance['overall_compliant']:
                compliance_results['overall_compliance'] = False
                
            compliance_results['violations'].extend(test_compliance['violations'])
        
        # Generate recommendations
        compliance_results['recommendations'] = self._generate_sla_recommendations(compliance_results)
        
        return compliance_results
    
    def _validate_single_test(self, test_result: TestSummary) -> Dict[str, Any]:
        """Validate SLA compliance for single test"""
        violations = []
        compliance = {
            'availability': True,
            'response_time_p95': True,
            'response_time_p99': True,
            'error_rate': True,
            'throughput': True
        }
        
        # Calculate availability (inverse of error rate)
        error_rate = (test_result.failed_requests / test_result.total_requests) * 100 if test_result.total_requests > 0 else 0
        availability = 100 - error_rate
        
        # Check availability
        if availability < self.sla_targets['availability']:
            violations.append({
                'metric': 'availability',
                'target': self.sla_targets['availability'],
                'actual': availability,
                'severity': 'high'
            })
            compliance['availability'] = False
        
        # Check response times
        if test_result.p95_response_time > self.sla_targets['response_time_p95']:
            violations.append({
                'metric': 'response_time_p95',
                'target': self.sla_targets['response_time_p95'],
                'actual': test_result.p95_response_time,
                'severity': 'medium'
            })
            compliance['response_time_p95'] = False
        
        if test_result.p99_response_time > self.sla_targets['response_time_p99']:
            violations.append({
                'metric': 'response_time_p99',
                'target': self.sla_targets['response_time_p99'],
                'actual': test_result.p99_response_time,
                'severity': 'medium'
            })
            compliance['response_time_p99'] = False
        
        # Check error rate
        if error_rate > self.sla_targets['error_rate']:
            violations.append({
                'metric': 'error_rate',
                'target': self.sla_targets['error_rate'],
                'actual': error_rate,
                'severity': 'high'
            })
            compliance['error_rate'] = False
        
        # Check throughput
        if test_result.requests_per_second < self.sla_targets['throughput_min']:
            violations.append({
                'metric': 'throughput',
                'target': self.sla_targets['throughput_min'],
                'actual': test_result.requests_per_second,
                'severity': 'medium'
            })
            compliance['throughput'] = False
        
        return {
            'overall_compliant': all(compliance.values()),
            'individual_compliance': compliance,
            'violations': violations,
            'metrics': {
                'availability': availability,
                'response_time_p95': test_result.p95_response_time,
                'response_time_p99': test_result.p99_response_time,
                'error_rate': error_rate,
                'throughput': test_result.requests_per_second
            }
        }
    
    def _generate_sla_recommendations(self, compliance_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations to improve SLA compliance"""
        recommendations = []
        
        # Analyze violations
        violation_counts = {}
        for violation in compliance_results['violations']:
            metric = violation['metric']
            violation_counts[metric] = violation_counts.get(metric, 0) + 1
        
        # Generate specific recommendations
        for metric, count in violation_counts.items():
            if metric == 'availability':
                recommendations.append({
                    'metric': metric,
                    'issue': f"Availability violations in {count} tests",
                    'recommendation': "Implement circuit breakers and improve error handling",
                    'priority': 'high'
                })
            elif metric == 'response_time_p95':
                recommendations.append({
                    'metric': metric,
                    'issue': f"P95 response time violations in {count} tests",
                    'recommendation': "Optimize application performance and increase resources",
                    'priority': 'medium'
                })
            elif metric == 'error_rate':
                recommendations.append({
                    'metric': metric,
                    'issue': f"Error rate violations in {count} tests",
                    'recommendation': "Implement better error handling and input validation",
                    'priority': 'high'
                })
            elif metric == 'throughput':
                recommendations.append({
                    'metric': metric,
                    'issue': f"Throughput violations in {count} tests",
                    'recommendation': "Scale horizontally or optimize request processing",
                    'priority': 'medium'
                })
        
        return recommendations

async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='PNO Physics Bench Performance Testing Suite')
    parser.add_argument('--base-url', default='https://api.pno-physics-bench.com', 
                       help='Base URL for testing')
    parser.add_argument('--test-type', choices=['load', 'capacity', 'sla', 'all'], 
                       default='all', help='Type of test to run')
    parser.add_argument('--output-dir', default='/tmp/performance-results', 
                       help='Output directory for results')
    parser.add_argument('--concurrent-users', type=int, default=50, 
                       help='Number of concurrent users for load test')
    parser.add_argument('--duration', type=int, default=300, 
                       help='Test duration in seconds')
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize testing components
    tester = PerformanceTester(args.base_url)
    capacity_planner = CapacityPlanner(tester)
    sla_validator = SLAValidator()
    
    results = {}
    
    try:
        if args.test_type in ['load', 'all']:
            logger.info("Running load tests")
            
            # Basic load test
            load_config = LoadTestConfig(
                url=f"{args.base_url}/predict",
                method="POST",
                concurrent_users=args.concurrent_users,
                duration_seconds=args.duration,
                ramp_up_seconds=30,
                payload={
                    "input_data": [[1.0, 2.0, 3.0, 4.0, 5.0]] * 100,
                    "model_config": {"uncertainty": True}
                }
            )
            
            load_results = await tester.run_load_test(load_config)
            results['load_test'] = {
                'configuration': asdict(load_config),
                'results': asdict(load_results)
            }
            
            logger.info(f"Load test completed: {load_results.requests_per_second:.1f} RPS, "
                       f"P95: {load_results.p95_response_time:.1f}ms")
        
        if args.test_type in ['capacity', 'all']:
            logger.info("Running capacity analysis")
            capacity_results = await capacity_planner.run_capacity_analysis(args.base_url)
            results['capacity_analysis'] = capacity_results
            
            logger.info("Capacity analysis completed")
        
        if args.test_type in ['sla', 'all']:
            logger.info("Running SLA validation")
            
            # Run multiple tests for SLA validation
            test_summaries = []
            
            # Quick tests with different loads
            test_configs = [
                LoadTestConfig(url=f"{args.base_url}/health", concurrent_users=10, duration_seconds=60),
                LoadTestConfig(url=f"{args.base_url}/predict", method="POST", 
                              concurrent_users=25, duration_seconds=120,
                              payload={"input_data": [[1.0, 2.0]] * 50}),
                LoadTestConfig(url=f"{args.base_url}/predict", method="POST",
                              concurrent_users=50, duration_seconds=180,
                              payload={"input_data": [[1.0, 2.0]] * 100})
            ]
            
            for config in test_configs:
                summary = await tester.run_load_test(config)
                test_summaries.append(summary)
            
            sla_results = sla_validator.validate_sla_compliance(test_summaries)
            results['sla_validation'] = sla_results
            
            logger.info(f"SLA validation completed: {'COMPLIANT' if sla_results['overall_compliance'] else 'NON-COMPLIANT'}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{args.output_dir}/performance_test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Generate summary report
        generate_summary_report(results, f"{args.output_dir}/performance_summary_{timestamp}.md")
        
        # Generate capacity planning report
        if 'capacity_analysis' in results:
            generate_capacity_report(results['capacity_analysis'], 
                                   f"{args.output_dir}/capacity_report_{timestamp}.md")
        
        logger.info("Performance testing completed successfully")
        
    except Exception as e:
        logger.error(f"Performance testing failed: {e}")
        raise

def generate_summary_report(results: Dict[str, Any], output_path: str):
    """Generate human-readable summary report"""
    
    report = f"""# PNO Physics Bench - Performance Test Summary

## Test Execution Summary
**Timestamp**: {datetime.now().isoformat()}
**Test Types**: {', '.join(results.keys())}

"""
    
    if 'load_test' in results:
        load_data = results['load_test']['results']
        report += f"""## Load Test Results

**Configuration**:
- Concurrent Users: {results['load_test']['configuration']['concurrent_users']}
- Duration: {results['load_test']['configuration']['duration_seconds']}s
- Endpoint: {results['load_test']['configuration']['url']}

**Performance Metrics**:
- Total Requests: {load_data['total_requests']:,}
- Successful Requests: {load_data['successful_requests']:,}
- Failed Requests: {load_data['failed_requests']:,}
- Requests per Second: {load_data['requests_per_second']:.1f}
- Average Response Time: {load_data['avg_response_time']:.1f}ms
- P95 Response Time: {load_data['p95_response_time']:.1f}ms
- P99 Response Time: {load_data['p99_response_time']:.1f}ms

"""
    
    if 'sla_validation' in results:
        sla_data = results['sla_validation']
        compliance_status = "✅ COMPLIANT" if sla_data['overall_compliance'] else "❌ NON-COMPLIANT"
        
        report += f"""## SLA Compliance Validation

**Overall Status**: {compliance_status}

**SLA Targets**:
- Availability: {sla_data['sla_targets']['availability']}%
- P95 Response Time: {sla_data['sla_targets']['response_time_p95']}ms
- P99 Response Time: {sla_data['sla_targets']['response_time_p99']}ms
- Error Rate: {sla_data['sla_targets']['error_rate']}%
- Minimum Throughput: {sla_data['sla_targets']['throughput_min']} RPS

"""
        
        if sla_data['violations']:
            report += "**Violations**:\n"
            for violation in sla_data['violations']:
                report += f"- {violation['metric']}: Target {violation['target']}, Actual {violation['actual']:.2f}\n"
    
    if 'capacity_analysis' in results:
        capacity_data = results['capacity_analysis']
        recommendations = capacity_data.get('recommendations', {})
        
        report += f"""## Capacity Planning Summary

**Stress Test Results**:
- Maximum RPS Tested: {capacity_data.get('stress_test', {}).get('system_limits', {}).get('max_rps', 'N/A')}
- Maximum Concurrent Users: {capacity_data.get('stress_test', {}).get('system_limits', {}).get('max_concurrent_users', 'N/A')}

**Scaling Recommendations**:
"""
        
        for rec in recommendations.get('scaling_recommendations', []):
            report += f"- {rec.get('recommendation', 'No recommendations available')}\n"
    
    with open(output_path, 'w') as f:
        f.write(report)

def generate_capacity_report(capacity_data: Dict[str, Any], output_path: str):
    """Generate detailed capacity planning report"""
    
    report = f"""# PNO Physics Bench - Capacity Planning Report

## Executive Summary
**Analysis Date**: {capacity_data.get('timestamp', 'Unknown')}

This report provides detailed capacity planning analysis for the PNO Physics Bench production environment.

## Test Results Summary

### Baseline Performance
"""
    
    baseline = capacity_data.get('baseline_performance', {})
    if baseline:
        baseline_results = baseline.get('results', {})
        report += f"""
- Average Response Time: {baseline_results.get('avg_response_time', 0):.1f}ms
- P95 Response Time: {baseline_results.get('p95_response_time', 0):.1f}ms
- Baseline RPS: {baseline_results.get('requests_per_second', 0):.1f}
"""
    
    # Load progression analysis
    progression = capacity_data.get('load_progression', [])
    if progression:
        report += """
### Load Progression Analysis

| Users | RPS | P95 Response Time | Error Rate | Status |
|-------|-----|-------------------|------------|--------|
"""
        
        for result in progression:
            users = result['concurrent_users']
            rps = result['results']['requests_per_second']
            p95 = result['results']['p95_response_time']
            error_rate = (result['results']['failed_requests'] / 
                         result['results']['total_requests']) * 100 if result['results']['total_requests'] > 0 else 0
            status = "✅ Good" if error_rate < 1 and p95 < 500 else "⚠️ Degraded" if error_rate < 5 else "❌ Poor"
            
            report += f"| {users} | {rps:.1f} | {p95:.1f}ms | {error_rate:.2f}% | {status} |\n"
    
    # Recommendations
    recommendations = capacity_data.get('recommendations', {})
    if recommendations:
        report += """
## Recommendations

### Scaling Recommendations
"""
        for rec in recommendations.get('scaling_recommendations', []):
            report += f"- **{rec.get('metric', 'Unknown')}**: {rec.get('recommendation', 'No recommendation')}\n"
        
        report += """
### Resource Recommendations
"""
        for rec in recommendations.get('resource_recommendations', []):
            report += f"- **{rec.get('metric', 'Unknown')}**: {rec.get('recommendation', 'No recommendation')}\n"
    
    with open(output_path, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    asyncio.run(main())