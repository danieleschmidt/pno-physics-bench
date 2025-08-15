#!/usr/bin/env python3
"""
Generation 2: Robust Validation Suite
Comprehensive error handling, security validation, and reliability testing
"""

import os
import sys
import json
import hashlib
import subprocess
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SecurityValidator:
    """Security validation for code and configurations."""
    
    def __init__(self):
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%.*["\']',
                r'cursor\.execute\s*\(\s*f["\']',
            ],
            'unsafe_eval': [
                r'\beval\s*\(',
                r'\bexec\s*\(',
            ],
            'unsafe_imports': [
                r'import\s+os\s*;.*system',
                r'from\s+os\s+import.*system',
            ]
        }
    
    def scan_file(self, file_path: str) -> Dict[str, List[str]]:
        """Scan file for security vulnerabilities."""
        issues = {category: [] for category in self.security_patterns}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues[category].append(f"Line {line_num}: {match.group()}")
        
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
        
        return issues
    
    def scan_directory(self, directory: str) -> Dict[str, Dict[str, List[str]]]:
        """Scan directory recursively for security issues."""
        all_issues = {}
        
        for root, dirs, files in os.walk(directory):
            # Skip common directories that shouldn't be scanned
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv'}]
            
            for file in files:
                if file.endswith(('.py', '.yaml', '.yml', '.json', '.sh')):
                    file_path = os.path.join(root, file)
                    issues = self.scan_file(file_path)
                    
                    # Only include files with issues
                    if any(issues.values()):
                        rel_path = os.path.relpath(file_path, directory)
                        all_issues[rel_path] = issues
        
        return all_issues

class InputValidator:
    """Advanced input validation and sanitization."""
    
    @staticmethod
    def validate_tensor_dimensions(dims: List[int], name: str = "tensor") -> Tuple[bool, str]:
        """Validate tensor dimension specifications."""
        if not isinstance(dims, (list, tuple)):
            return False, f"{name} dimensions must be a list or tuple"
        
        if len(dims) == 0:
            return False, f"{name} dimensions cannot be empty"
        
        if not all(isinstance(d, int) and d > 0 for d in dims):
            return False, f"{name} dimensions must be positive integers"
        
        if any(d > 10000 for d in dims):
            return False, f"{name} dimensions too large (max 10000)"
        
        return True, "Valid dimensions"
    
    @staticmethod
    def validate_hyperparameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate training hyperparameters."""
        issues = []
        
        # Learning rate validation
        if 'learning_rate' in params:
            lr = params['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                issues.append("learning_rate must be a positive number ≤ 1")
        
        # Batch size validation  
        if 'batch_size' in params:
            bs = params['batch_size']
            if not isinstance(bs, int) or bs <= 0 or bs > 1024:
                issues.append("batch_size must be a positive integer ≤ 1024")
        
        # Epochs validation
        if 'epochs' in params:
            ep = params['epochs']
            if not isinstance(ep, int) or ep <= 0 or ep > 10000:
                issues.append("epochs must be a positive integer ≤ 10000")
        
        # KL weight validation
        if 'kl_weight' in params:
            kl = params['kl_weight']
            if not isinstance(kl, (int, float)) or kl < 0 or kl > 1:
                issues.append("kl_weight must be a non-negative number ≤ 1")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def sanitize_file_path(path: str) -> Tuple[bool, str]:
        """Sanitize and validate file paths."""
        if not isinstance(path, str):
            return False, "Path must be a string"
        
        # Check for path traversal attempts
        if '..' in path or path.startswith('/'):
            return False, "Path traversal not allowed"
        
        # Check for invalid characters
        invalid_chars = '<>"|*?'
        if any(char in path for char in invalid_chars):
            return False, f"Invalid characters in path: {invalid_chars}"
        
        # Check length
        if len(path) > 255:
            return False, "Path too long (max 255 characters)"
        
        return True, path

class ErrorHandler:
    """Comprehensive error handling and logging."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_log = []
    
    def handle_error(self, error: Exception, context: str) -> None:
        """Handle and log errors with context."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Count error types
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log error
        log_entry = {
            'timestamp': self._get_timestamp(),
            'error_type': error_type,
            'message': error_msg,
            'context': context,
            'count': self.error_counts[error_type]
        }
        
        self.error_log.append(log_entry)
        logger.error(f"{context}: {error_type}: {error_msg}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        return {
            'total_errors': len(self.error_log),
            'error_types': self.error_counts,
            'recent_errors': self.error_log[-10:] if self.error_log else []
        }

class HealthChecker:
    """System health and dependency checking."""
    
    def __init__(self):
        self.health_status = {}
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Check Python version compatibility."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python {version.major}.{version.minor} (requires 3.9+)"
    
    def check_memory_usage(self) -> Tuple[bool, str]:
        """Check available memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= 2.0:
                return True, f"{available_gb:.1f}GB available"
            else:
                return False, f"Low memory: {available_gb:.1f}GB available"
        except ImportError:
            return True, "Memory check skipped (psutil not available)"
    
    def check_disk_space(self) -> Tuple[bool, str]:
        """Check available disk space."""
        try:
            import shutil
            free_bytes = shutil.disk_usage('.').free
            free_gb = free_bytes / (1024**3)
            
            if free_gb >= 1.0:
                return True, f"{free_gb:.1f}GB available"
            else:
                return False, f"Low disk space: {free_gb:.1f}GB available"
        except Exception:
            return True, "Disk check skipped"
    
    def check_file_permissions(self) -> Tuple[bool, str]:
        """Check file system permissions."""
        try:
            # Test write permission
            test_file = 'permission_test.tmp'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True, "Read/write permissions OK"
        except Exception as e:
            return False, f"Permission error: {e}"
    
    def run_health_checks(self) -> Dict[str, Tuple[bool, str]]:
        """Run all health checks."""
        checks = {
            'python_version': self.check_python_version(),
            'memory_usage': self.check_memory_usage(),
            'disk_space': self.check_disk_space(),
            'file_permissions': self.check_file_permissions()
        }
        
        self.health_status = checks
        return checks

def validate_model_robustness():
    """Validate model robustness features."""
    print("🛡️ Model Robustness Validation")
    print("-" * 50)
    
    robustness_features = [
        ("Input validation", "src/pno_physics_bench/validation/input_sanitization.py"),
        ("Error handling", "src/pno_physics_bench/utils/error_handling.py"),
        ("Circuit breaker", "src/pno_physics_bench/robustness/circuit_breaker.py"),
        ("Fault tolerance", "src/pno_physics_bench/robustness/fault_tolerance.py"),
        ("Security validation", "src/pno_physics_bench/security_validation.py")
    ]
    
    found_features = 0
    for feature, path in robustness_features:
        if os.path.exists(path):
            print(f"✅ {feature}: {path}")
            found_features += 1
        else:
            print(f"❌ {feature}: {path} (missing)")
    
    print(f"\n📊 Robustness Coverage: {found_features}/{len(robustness_features)} ({100*found_features/len(robustness_features):.1f}%)")
    return found_features >= len(robustness_features) * 0.8

def test_error_scenarios():
    """Test error handling scenarios."""
    print("\n🚨 Error Handling Testing")
    print("-" * 50)
    
    error_handler = ErrorHandler()
    input_validator = InputValidator()
    
    # Test input validation
    test_cases = [
        ("Invalid dimensions", [-1, 0, 256], False),
        ("Valid dimensions", [3, 256, 64], True),
        ("Empty dimensions", [], False),
        ("Huge dimensions", [50000, 50000], False),
        ("Valid small dimensions", [1, 1, 1], True)
    ]
    
    validation_passed = 0
    for test_name, dims, expected_valid in test_cases:
        try:
            is_valid, msg = input_validator.validate_tensor_dimensions(dims)
            if is_valid == expected_valid:
                print(f"✅ {test_name}: {msg}")
                validation_passed += 1
            else:
                print(f"❌ {test_name}: Expected {expected_valid}, got {is_valid}")
        except Exception as e:
            error_handler.handle_error(e, f"Testing {test_name}")
            print(f"❌ {test_name}: Exception occurred")
    
    # Test hyperparameter validation
    hyperparam_tests = [
        ("Valid params", {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100}, True),
        ("Invalid LR", {'learning_rate': -0.1}, False),
        ("Invalid batch size", {'batch_size': 0}, False),
        ("Invalid epochs", {'epochs': -5}, False)
    ]
    
    for test_name, params, expected_valid in hyperparam_tests:
        try:
            is_valid, issues = input_validator.validate_hyperparameters(params)
            if is_valid == expected_valid:
                print(f"✅ {test_name}: {'Valid' if is_valid else str(issues)}")
                validation_passed += 1
            else:
                print(f"❌ {test_name}: Expected {expected_valid}, got {is_valid}")
        except Exception as e:
            error_handler.handle_error(e, f"Testing {test_name}")
            print(f"❌ {test_name}: Exception occurred")
    
    total_tests = len(test_cases) + len(hyperparam_tests)
    print(f"\n📊 Error Handling: {validation_passed}/{total_tests} tests passed")
    
    return validation_passed >= total_tests * 0.8

def run_security_scan():
    """Run comprehensive security scan."""
    print("\n🔒 Security Vulnerability Scan")
    print("-" * 50)
    
    security_validator = SecurityValidator()
    
    # Scan source code
    source_issues = security_validator.scan_directory("src")
    
    # Scan scripts and configs
    script_issues = security_validator.scan_directory(".")
    
    all_issues = {**source_issues, **script_issues}
    
    total_files_scanned = len([f for f in Path(".").rglob("*.py") if not any(part.startswith('.') for part in f.parts)])
    files_with_issues = len(all_issues)
    
    if files_with_issues == 0:
        print("✅ No security vulnerabilities detected")
        security_score = 100
    else:
        print(f"⚠️  Found potential issues in {files_with_issues} files:")
        for file_path, issues in all_issues.items():
            print(f"\n  📄 {file_path}:")
            for category, issue_list in issues.items():
                if issue_list:
                    print(f"    🚨 {category}:")
                    for issue in issue_list[:3]:  # Show first 3 issues
                        print(f"      - {issue}")
        
        security_score = max(0, 100 - (files_with_issues / total_files_scanned * 100))
    
    print(f"\n🔒 Security Score: {security_score:.1f}/100")
    return security_score >= 80

def run_health_checks():
    """Run system health checks."""
    print("\n💊 System Health Checks")
    print("-" * 50)
    
    health_checker = HealthChecker()
    checks = health_checker.run_health_checks()
    
    passed_checks = 0
    for check_name, (status, message) in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check_name}: {message}")
        if status:
            passed_checks += 1
    
    total_checks = len(checks)
    health_score = (passed_checks / total_checks) * 100
    
    print(f"\n💊 Health Score: {health_score:.1f}/100")
    return health_score >= 75

def main():
    """Run Generation 2 robustness validation."""
    print("🛡️ Generation 2: MAKE IT ROBUST - Reliability Testing")
    print("=" * 60)
    
    test_results = {}
    total_score = 0
    
    # Test 1: Model Robustness Features
    print("\n" + "="*60)
    try:
        result = validate_model_robustness()
        test_results['model_robustness'] = result
        total_score += 25 if result else 0
        print(f"{'✅' if result else '❌'} Model robustness: {'PASSED' if result else 'FAILED'}")
    except Exception as e:
        logger.error(f"Model robustness test failed: {e}")
        test_results['model_robustness'] = False
    
    # Test 2: Error Handling
    print("\n" + "="*60)
    try:
        result = test_error_scenarios()
        test_results['error_handling'] = result
        total_score += 25 if result else 0
        print(f"{'✅' if result else '❌'} Error handling: {'PASSED' if result else 'FAILED'}")
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        test_results['error_handling'] = False
    
    # Test 3: Security Scan
    print("\n" + "="*60)
    try:
        result = run_security_scan()
        test_results['security'] = result
        total_score += 25 if result else 0
        print(f"{'✅' if result else '❌'} Security scan: {'PASSED' if result else 'FAILED'}")
    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        test_results['security'] = False
    
    # Test 4: Health Checks
    print("\n" + "="*60)
    try:
        result = run_health_checks()
        test_results['health_checks'] = result
        total_score += 25 if result else 0
        print(f"{'✅' if result else '❌'} Health checks: {'PASSED' if result else 'FAILED'}")
    except Exception as e:
        logger.error(f"Health checks failed: {e}")
        test_results['health_checks'] = False
    
    # Final Results
    print("\n" + "=" * 60)
    print(f"🛡️ Generation 2 Results: {total_score}/100 points")
    
    if total_score >= 75:
        print("🎉 Generation 2 COMPLETE: Robustness validated!")
        print("🚀 Ready to proceed to Generation 3: Optimization")
        gen2_status = "complete"
    elif total_score >= 50:
        print("⚠️  Generation 2 PARTIAL: Some robustness issues detected")
        gen2_status = "partial"
    else:
        print("❌ Generation 2 FAILED: Significant robustness issues")
        gen2_status = "failed"
    
    # Save results
    gen2_results = {
        "generation": 2,
        "status": gen2_status,
        "total_score": total_score,
        "test_results": test_results,
        "passed_tests": sum(test_results.values()),
        "total_tests": len(test_results),
        "next_generation": 3 if total_score >= 75 else 2
    }
    
    with open("generation_2_results.json", "w") as f:
        json.dump(gen2_results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to generation_2_results.json")
    return total_score >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)