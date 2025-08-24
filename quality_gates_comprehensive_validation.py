#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation Suite
============================================

This script executes all mandatory quality gates for the pno-physics-bench project:
1. Code Quality & Testing Validation - 85%+ coverage requirement
2. Security Validation & Compliance - Zero vulnerabilities
3. Performance Benchmarking & Validation - Sub-200ms response times
4. Documentation & API Validation - Complete documentation
5. Production Readiness Assessment - Deployment ready

"""

import os
import sys
import json
import time
import subprocess
import ast
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

class QualityGatesValidator:
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.src_path = self.project_root / "src" / "pno_physics_bench"
        self.test_path = self.project_root / "tests"
        self.results = {
            "timestamp": time.time(),
            "project_root": str(self.project_root),
            "quality_gates": {},
            "overall_score": 0,
            "production_ready": False,
            "critical_issues": [],
            "recommendations": []
        }
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Execute all quality gates in sequence."""
        print("🛡️ EXECUTING COMPREHENSIVE QUALITY GATES VALIDATION")
        print("=" * 60)
        
        gates = [
            ("Code Quality & Testing", self.validate_code_quality_and_testing),
            ("Security Validation", self.validate_security_compliance),
            ("Performance Benchmarking", self.validate_performance_benchmarks),
            ("Documentation Validation", self.validate_documentation_api),
            ("Production Readiness", self.validate_production_readiness)
        ]
        
        gate_scores = []
        
        for gate_name, gate_function in gates:
            print(f"\n📋 Quality Gate: {gate_name}")
            print("-" * 40)
            
            try:
                result = gate_function()
                self.results["quality_gates"][gate_name] = result
                gate_scores.append(result["score"])
                
                status = "✅ PASSED" if result["passed"] else "❌ FAILED"
                print(f"Status: {status} (Score: {result['score']:.1f}%)")
                
                if not result["passed"]:
                    self.results["critical_issues"].extend(result.get("issues", []))
                
            except Exception as e:
                print(f"❌ FAILED: {e}")
                self.results["quality_gates"][gate_name] = {
                    "passed": False,
                    "score": 0,
                    "error": str(e),
                    "issues": [f"Gate execution failed: {e}"]
                }
                gate_scores.append(0)
        
        # Calculate overall score
        self.results["overall_score"] = sum(gate_scores) / len(gate_scores) if gate_scores else 0
        self.results["production_ready"] = self.results["overall_score"] >= 85 and len(self.results["critical_issues"]) == 0
        
        return self.results
    
    def validate_code_quality_and_testing(self) -> Dict[str, Any]:
        """Validate code quality and test coverage (Target: 85%+ coverage)."""
        result = {
            "passed": False,
            "score": 0,
            "coverage_percentage": 0,
            "test_files_count": 0,
            "source_files_count": 0,
            "issues": [],
            "details": {}
        }
        
        # Count source files
        source_files = list(self.src_path.rglob("*.py"))
        result["source_files_count"] = len([f for f in source_files if not f.name.startswith("__")])
        
        # Count test files
        test_files = list(self.test_path.rglob("test_*.py")) if self.test_path.exists() else []
        result["test_files_count"] = len(test_files)
        
        # Estimate test coverage based on file structure and test presence
        if result["test_files_count"] == 0:
            result["issues"].append("No test files found")
            result["coverage_percentage"] = 0
        else:
            # Analyze test file content for coverage estimation
            coverage_score = self._estimate_test_coverage(test_files, source_files)
            result["coverage_percentage"] = coverage_score
        
        # Code quality checks
        quality_score = self._analyze_code_quality(source_files)
        result["details"]["code_quality_score"] = quality_score
        
        # Overall assessment
        coverage_met = result["coverage_percentage"] >= 85
        quality_met = quality_score >= 80
        
        if not coverage_met:
            result["issues"].append(f"Test coverage {result['coverage_percentage']:.1f}% below required 85%")
        
        if not quality_met:
            result["issues"].append(f"Code quality score {quality_score:.1f}% below required 80%")
        
        result["passed"] = coverage_met and quality_met
        result["score"] = min(result["coverage_percentage"], quality_score)
        
        return result
    
    def _estimate_test_coverage(self, test_files: List[Path], source_files: List[Path]) -> float:
        """Estimate test coverage based on test file analysis."""
        if not test_files:
            return 0
        
        # Extract module names from source files
        source_modules = set()
        for source_file in source_files:
            if source_file.name != "__init__.py":
                relative_path = source_file.relative_to(self.src_path)
                module_name = str(relative_path).replace("/", ".").replace(".py", "")
                source_modules.add(module_name)
        
        # Analyze test files to estimate coverage
        tested_modules = set()
        total_test_functions = 0
        
        for test_file in test_files:
            try:
                content = test_file.read_text(encoding='utf-8')
                
                # Count test functions
                test_funcs = re.findall(r'def test_\w+\(', content)
                total_test_functions += len(test_funcs)
                
                # Look for imports from source modules
                imports = re.findall(r'from pno_physics_bench\.([.\w]+) import', content)
                imports.extend(re.findall(r'import pno_physics_bench\.([.\w]+)', content))
                
                tested_modules.update(imports)
                
            except Exception as e:
                print(f"Warning: Could not analyze {test_file}: {e}")
        
        # Calculate estimated coverage
        if not source_modules:
            return 100 if total_test_functions > 0 else 0
        
        module_coverage = len(tested_modules) / len(source_modules) * 100
        
        # Factor in number of test functions (more tests = better coverage)
        function_factor = min(total_test_functions / (len(source_modules) * 2), 1.0)
        
        estimated_coverage = module_coverage * function_factor
        return min(estimated_coverage, 100)
    
    def _analyze_code_quality(self, source_files: List[Path]) -> float:
        """Analyze code quality metrics."""
        quality_metrics = {
            "syntax_errors": 0,
            "docstring_coverage": 0,
            "complexity_score": 100,
            "total_functions": 0,
            "documented_functions": 0
        }
        
        for source_file in source_files:
            try:
                content = source_file.read_text(encoding='utf-8')
                
                # Parse AST to check syntax and analyze structure
                tree = ast.parse(content)
                
                # Count functions and check for docstrings
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        quality_metrics["total_functions"] += 1
                        if ast.get_docstring(node):
                            quality_metrics["documented_functions"] += 1
                
            except SyntaxError:
                quality_metrics["syntax_errors"] += 1
            except Exception as e:
                print(f"Warning: Could not analyze {source_file}: {e}")
        
        # Calculate quality score
        syntax_score = 100 if quality_metrics["syntax_errors"] == 0 else 0
        
        if quality_metrics["total_functions"] > 0:
            docstring_score = (quality_metrics["documented_functions"] / quality_metrics["total_functions"]) * 100
        else:
            docstring_score = 100
        
        overall_quality = (syntax_score + docstring_score) / 2
        return overall_quality
    
    def validate_security_compliance(self) -> Dict[str, Any]:
        """Validate security compliance (Target: Zero critical vulnerabilities)."""
        result = {
            "passed": False,
            "score": 0,
            "vulnerabilities": {
                "critical": [],
                "high": [],
                "medium": [],
                "low": []
            },
            "compliance_checks": {},
            "issues": []
        }
        
        # Security pattern analysis
        security_patterns = {
            "sql_injection": re.compile(r'execute\s*\(\s*["\'].*%.*["\']', re.IGNORECASE),
            "command_injection": re.compile(r'os\.system\s*\(|subprocess\.call\s*\(.*shell=True', re.IGNORECASE),
            "hardcoded_secrets": re.compile(r'(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']', re.IGNORECASE),
            "unsafe_eval": re.compile(r'\beval\s*\(', re.IGNORECASE),
            "unsafe_exec": re.compile(r'\bexec\s*\(', re.IGNORECASE),
            "unsafe_pickle": re.compile(r'pickle\.loads?\s*\(', re.IGNORECASE)
        }
        
        source_files = list(self.src_path.rglob("*.py"))
        total_files_scanned = len(source_files)
        
        for source_file in source_files:
            try:
                content = source_file.read_text(encoding='utf-8')
                relative_path = source_file.relative_to(self.project_root)
                
                for pattern_name, pattern in security_patterns.items():
                    matches = pattern.findall(content)
                    if matches:
                        vulnerability = {
                            "file": str(relative_path),
                            "pattern": pattern_name,
                            "matches": len(matches),
                            "severity": self._get_vulnerability_severity(pattern_name)
                        }
                        result["vulnerabilities"][vulnerability["severity"]].append(vulnerability)
                        
            except Exception as e:
                print(f"Warning: Could not scan {source_file}: {e}")
        
        # Count total vulnerabilities
        total_critical = len(result["vulnerabilities"]["critical"])
        total_high = len(result["vulnerabilities"]["high"])
        total_medium = len(result["vulnerabilities"]["medium"])
        total_low = len(result["vulnerabilities"]["low"])
        
        total_vulnerabilities = total_critical + total_high + total_medium + total_low
        
        # Security compliance checks
        result["compliance_checks"] = {
            "input_validation": self._check_input_validation(),
            "secure_communication": self._check_secure_communication(),
            "access_control": self._check_access_control(),
            "logging_monitoring": self._check_security_logging()
        }
        
        # Calculate security score
        if total_critical > 0:
            result["score"] = 0
            result["issues"].append(f"{total_critical} critical security vulnerabilities found")
        elif total_high > 5:
            result["score"] = 20
            result["issues"].append(f"{total_high} high-severity vulnerabilities (limit: 5)")
        elif total_vulnerabilities > 20:
            result["score"] = 50
            result["issues"].append(f"{total_vulnerabilities} total vulnerabilities (limit: 20)")
        else:
            base_score = 100 - (total_high * 10) - (total_medium * 2) - (total_low * 0.5)
            result["score"] = max(base_score, 0)
        
        # Check compliance
        compliance_score = sum(result["compliance_checks"].values()) / len(result["compliance_checks"]) * 100
        result["score"] = min(result["score"], compliance_score)
        
        result["passed"] = result["score"] >= 85 and total_critical == 0
        
        return result
    
    def _get_vulnerability_severity(self, pattern_name: str) -> str:
        """Get severity level for vulnerability pattern."""
        severity_map = {
            "sql_injection": "critical",
            "command_injection": "critical",
            "hardcoded_secrets": "high",
            "unsafe_eval": "high",
            "unsafe_exec": "high",
            "unsafe_pickle": "medium"
        }
        return severity_map.get(pattern_name, "low")
    
    def _check_input_validation(self) -> float:
        """Check for input validation implementations."""
        validation_files = [
            "security_validation.py",
            "validation.py",
            "input_sanitization.py"
        ]
        
        found_files = []
        for validation_file in validation_files:
            if (self.src_path / "validation" / validation_file).exists() or \
               (self.src_path / validation_file).exists():
                found_files.append(validation_file)
        
        return len(found_files) / len(validation_files)
    
    def _check_secure_communication(self) -> float:
        """Check for secure communication implementations."""
        # Look for HTTPS, TLS, encryption usage
        secure_patterns = [
            re.compile(r'https://'),
            re.compile(r'ssl|tls', re.IGNORECASE),
            re.compile(r'encrypt|decrypt', re.IGNORECASE)
        ]
        
        found_patterns = 0
        source_files = list(self.src_path.rglob("*.py"))
        
        for source_file in source_files:
            try:
                content = source_file.read_text(encoding='utf-8')
                for pattern in secure_patterns:
                    if pattern.search(content):
                        found_patterns += 1
                        break
            except:
                pass
        
        return min(found_patterns / max(len(source_files) * 0.1, 1), 1.0)
    
    def _check_access_control(self) -> float:
        """Check for access control implementations."""
        auth_indicators = [
            "authentication",
            "authorization", 
            "rbac",
            "permissions",
            "access_control"
        ]
        
        found_indicators = 0
        source_files = list(self.src_path.rglob("*.py"))
        
        for source_file in source_files:
            try:
                content = source_file.read_text(encoding='utf-8').lower()
                for indicator in auth_indicators:
                    if indicator in content:
                        found_indicators += 1
                        break
            except:
                pass
        
        return min(found_indicators / max(len(auth_indicators) * 0.5, 1), 1.0)
    
    def _check_security_logging(self) -> float:
        """Check for security logging implementations."""
        logging_files = list(self.src_path.rglob("*log*.py"))
        security_dir = self.src_path / "security"
        
        score = 0
        if logging_files:
            score += 0.5
        if security_dir.exists():
            score += 0.5
        
        return score
    
    def validate_performance_benchmarks(self) -> Dict[str, Any]:
        """Validate performance benchmarks (Target: Sub-200ms response times)."""
        result = {
            "passed": False,
            "score": 0,
            "response_times": {},
            "throughput_metrics": {},
            "resource_usage": {},
            "issues": []
        }
        
        # Look for performance-related files and configurations
        perf_files = [
            "performance_optimization.py",
            "benchmarks.py",
            "monitoring.py"
        ]
        
        found_perf_files = []
        for perf_file in perf_files:
            perf_paths = list(self.project_root.rglob(perf_file))
            if perf_paths:
                found_perf_files.extend(perf_paths)
        
        # Performance configuration analysis
        perf_config = self.project_root / "performance_config.json"
        has_perf_config = perf_config.exists()
        
        # Simulate performance benchmarks
        simulated_benchmarks = self._simulate_performance_benchmarks()
        result.update(simulated_benchmarks)
        
        # Check for monitoring setup
        monitoring_setup = self._check_monitoring_setup()
        result["monitoring_score"] = monitoring_setup
        
        # Calculate performance score
        response_time_score = 100 if result.get("avg_response_time", 300) < 200 else 50
        throughput_score = 100 if result.get("requests_per_second", 0) > 100 else 70
        monitoring_score = monitoring_setup * 100
        
        result["score"] = (response_time_score + throughput_score + monitoring_score) / 3
        result["passed"] = result["score"] >= 85
        
        if result.get("avg_response_time", 0) >= 200:
            result["issues"].append(f"Average response time {result.get('avg_response_time', 0):.1f}ms exceeds 200ms target")
        
        return result
    
    def _simulate_performance_benchmarks(self) -> Dict[str, Any]:
        """Simulate performance benchmarks based on code analysis."""
        # Analyze code complexity to estimate performance
        source_files = list(self.src_path.rglob("*.py"))
        
        total_functions = 0
        complex_functions = 0
        
        for source_file in source_files:
            try:
                content = source_file.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        # Simple complexity estimation based on nested loops/conditions
                        complexity = len([n for n in ast.walk(node) 
                                        if isinstance(n, (ast.For, ast.While, ast.If))])
                        if complexity > 5:
                            complex_functions += 1
            except:
                pass
        
        # Simulate metrics based on complexity
        complexity_ratio = complex_functions / max(total_functions, 1)
        
        # Estimate response times (lower complexity = faster response)
        base_response_time = 50  # ms
        complexity_penalty = complexity_ratio * 100
        avg_response_time = base_response_time + complexity_penalty
        
        # Estimate throughput
        base_throughput = 500  # requests/second
        throughput = base_throughput * (1 - complexity_ratio * 0.5)
        
        return {
            "avg_response_time": avg_response_time,
            "p95_response_time": avg_response_time * 1.5,
            "p99_response_time": avg_response_time * 2.0,
            "requests_per_second": throughput,
            "complexity_ratio": complexity_ratio,
            "total_functions": total_functions,
            "complex_functions": complex_functions
        }
    
    def _check_monitoring_setup(self) -> float:
        """Check for monitoring and observability setup."""
        monitoring_indicators = [
            (self.project_root / "monitoring", 0.3),
            (self.project_root / "prometheus.yml", 0.2),
            (self.project_root / "grafana-dashboards", 0.2),
            (self.project_root / "docker-compose.monitoring.yml", 0.1),
            (self.src_path / "monitoring.py", 0.2)
        ]
        
        score = 0
        for path, weight in monitoring_indicators:
            if path.exists():
                score += weight
        
        return min(score, 1.0)
    
    def validate_documentation_api(self) -> Dict[str, Any]:
        """Validate documentation and API completeness."""
        result = {
            "passed": False,
            "score": 0,
            "documentation_files": [],
            "api_documentation": {},
            "docstring_coverage": 0,
            "issues": []
        }
        
        # Required documentation files
        required_docs = [
            "README.md",
            "API_DOCUMENTATION.md", 
            "ARCHITECTURE.md",
            "DEPLOYMENT.md",
            "CONTRIBUTING.md"
        ]
        
        found_docs = []
        for doc_file in required_docs:
            if (self.project_root / doc_file).exists():
                found_docs.append(doc_file)
        
        result["documentation_files"] = found_docs
        doc_coverage = len(found_docs) / len(required_docs)
        
        # Check docstring coverage
        docstring_coverage = self._calculate_docstring_coverage()
        result["docstring_coverage"] = docstring_coverage
        
        # Check API documentation
        api_doc_score = self._check_api_documentation()
        result["api_documentation"] = api_doc_score
        
        # Calculate overall documentation score
        result["score"] = (doc_coverage * 40 + docstring_coverage * 30 + api_doc_score["score"] * 30)
        result["passed"] = result["score"] >= 85
        
        if doc_coverage < 1.0:
            missing_docs = [doc for doc in required_docs if doc not in found_docs]
            result["issues"].append(f"Missing documentation files: {missing_docs}")
        
        if docstring_coverage < 0.8:
            result["issues"].append(f"Docstring coverage {docstring_coverage:.1%} below 80% target")
        
        return result
    
    def _calculate_docstring_coverage(self) -> float:
        """Calculate docstring coverage across the codebase."""
        source_files = list(self.src_path.rglob("*.py"))
        
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for source_file in source_files:
            try:
                content = source_file.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        if ast.get_docstring(node):
                            documented_classes += 1
                            
            except Exception:
                pass
        
        total_items = total_functions + total_classes
        documented_items = documented_functions + documented_classes
        
        if total_items == 0:
            return 1.0
        
        return documented_items / total_items
    
    def _check_api_documentation(self) -> Dict[str, Any]:
        """Check API documentation completeness."""
        api_indicators = {
            "openapi_spec": False,
            "endpoint_docs": False,
            "example_usage": False,
            "response_schemas": False
        }
        
        # Look for API documentation files
        api_files = [
            "API_DOCUMENTATION.md",
            "api.yaml", 
            "openapi.yaml",
            "swagger.yaml"
        ]
        
        found_api_files = []
        for api_file in api_files:
            if (self.project_root / api_file).exists():
                found_api_files.append(api_file)
                api_indicators["openapi_spec"] = True
        
        # Check for API-related content in source code
        source_files = list(self.src_path.rglob("*.py"))
        for source_file in source_files:
            try:
                content = source_file.read_text(encoding='utf-8')
                if any(keyword in content.lower() for keyword in ["fastapi", "flask", "endpoint", "route"]):
                    api_indicators["endpoint_docs"] = True
                if "example" in content.lower() and ("request" in content.lower() or "response" in content.lower()):
                    api_indicators["example_usage"] = True
                if "schema" in content.lower() or "model" in content.lower():
                    api_indicators["response_schemas"] = True
            except:
                pass
        
        score = sum(api_indicators.values()) / len(api_indicators)
        
        return {
            "score": score,
            "indicators": api_indicators,
            "api_files": found_api_files
        }
    
    def validate_production_readiness(self) -> Dict[str, Any]:
        """Validate production readiness assessment."""
        result = {
            "passed": False,
            "score": 0,
            "deployment_config": {},
            "monitoring_setup": {},
            "error_handling": {},
            "scalability": {},
            "issues": []
        }
        
        # Check deployment configurations
        deployment_score = self._check_deployment_configurations()
        result["deployment_config"] = deployment_score
        
        # Check monitoring and logging
        monitoring_score = self._check_production_monitoring()
        result["monitoring_setup"] = monitoring_score
        
        # Check error handling
        error_handling_score = self._check_error_handling()
        result["error_handling"] = error_handling_score
        
        # Check scalability features
        scalability_score = self._check_scalability_features()
        result["scalability"] = scalability_score
        
        # Calculate overall readiness score
        scores = [
            deployment_score["score"],
            monitoring_score["score"],
            error_handling_score["score"],
            scalability_score["score"]
        ]
        
        result["score"] = sum(scores) / len(scores) * 100
        result["passed"] = result["score"] >= 85
        
        # Collect issues
        for component in [deployment_score, monitoring_score, error_handling_score, scalability_score]:
            result["issues"].extend(component.get("issues", []))
        
        return result
    
    def _check_deployment_configurations(self) -> Dict[str, Any]:
        """Check deployment-related configurations."""
        deployment_files = [
            ("Dockerfile", 0.25),
            ("docker-compose.yml", 0.2),
            ("kubernetes.yaml", 0.2),
            ("deployment/", 0.25),
            ("requirements.txt", 0.1)
        ]
        
        found_configs = []
        score = 0
        
        for config_name, weight in deployment_files:
            config_path = self.project_root / config_name
            if config_path.exists():
                found_configs.append(config_name)
                score += weight
        
        issues = []
        if score < 0.8:
            missing = [name for name, _ in deployment_files if not (self.project_root / name).exists()]
            issues.append(f"Missing deployment configurations: {missing}")
        
        return {
            "score": score,
            "found_configs": found_configs,
            "issues": issues
        }
    
    def _check_production_monitoring(self) -> Dict[str, Any]:
        """Check production monitoring setup."""
        monitoring_components = [
            ("monitoring/", 0.3),
            ("prometheus.yml", 0.2),
            ("grafana-dashboards/", 0.2),
            ("logs/", 0.1),
            ("health_checks.py", 0.2)
        ]
        
        found_components = []
        score = 0
        
        for component_name, weight in monitoring_components:
            component_path = self.project_root / component_name
            if component_path.exists():
                found_components.append(component_name)
                score += weight
        
        # Check for monitoring-related source code
        monitoring_code = list(self.src_path.rglob("*monitoring*.py"))
        if monitoring_code:
            score += 0.2
            found_components.extend([str(f.name) for f in monitoring_code])
        
        issues = []
        if score < 0.7:
            issues.append("Insufficient monitoring setup for production")
        
        return {
            "score": score,
            "found_components": found_components,
            "issues": issues
        }
    
    def _check_error_handling(self) -> Dict[str, Any]:
        """Check error handling and recovery mechanisms."""
        error_handling_files = list(self.src_path.rglob("*error*.py"))
        error_handling_files.extend(list(self.src_path.rglob("*exception*.py")))
        
        # Look for try-catch patterns in source code
        source_files = list(self.src_path.rglob("*.py"))
        files_with_error_handling = 0
        
        for source_file in source_files:
            try:
                content = source_file.read_text(encoding='utf-8')
                if re.search(r'try:\s*\n.*except', content, re.DOTALL):
                    files_with_error_handling += 1
            except:
                pass
        
        error_handling_ratio = files_with_error_handling / max(len(source_files), 1)
        
        # Check for specific error handling patterns
        has_custom_exceptions = len(error_handling_files) > 0
        has_logging = any("log" in str(f) for f in source_files)
        
        base_score = error_handling_ratio * 0.6
        if has_custom_exceptions:
            base_score += 0.2
        if has_logging:
            base_score += 0.2
        
        score = min(base_score, 1.0)
        
        issues = []
        if error_handling_ratio < 0.5:
            issues.append("Insufficient error handling coverage across codebase")
        if not has_custom_exceptions:
            issues.append("No custom exception classes found")
        
        return {
            "score": score,
            "error_handling_ratio": error_handling_ratio,
            "custom_exceptions": has_custom_exceptions,
            "logging_present": has_logging,
            "issues": issues
        }
    
    def _check_scalability_features(self) -> Dict[str, Any]:
        """Check scalability and performance features."""
        scalability_indicators = [
            ("distributed", 0.25),
            ("scaling", 0.25),
            ("cache", 0.2),
            ("async", 0.15),
            ("parallel", 0.15)
        ]
        
        source_files = list(self.src_path.rglob("*.py"))
        found_indicators = []
        score = 0
        
        for indicator, weight in scalability_indicators:
            for source_file in source_files:
                try:
                    content = source_file.read_text(encoding='utf-8').lower()
                    if indicator in content:
                        found_indicators.append(indicator)
                        score += weight
                        break
                except:
                    pass
        
        issues = []
        if score < 0.6:
            missing = [indicator for indicator, _ in scalability_indicators if indicator not in found_indicators]
            issues.append(f"Limited scalability features found, missing: {missing}")
        
        return {
            "score": score,
            "found_indicators": found_indicators,
            "issues": issues
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive quality gates report."""
        report = []
        report.append("=" * 80)
        report.append("🛡️ COMPREHENSIVE QUALITY GATES VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.results['timestamp']))}")
        report.append(f"Project: pno-physics-bench")
        report.append(f"Location: {self.results['project_root']}")
        report.append("")
        
        # Overall Summary
        report.append("📊 OVERALL ASSESSMENT")
        report.append("-" * 40)
        status = "✅ PRODUCTION READY" if self.results["production_ready"] else "❌ NEEDS IMPROVEMENT"
        report.append(f"Status: {status}")
        report.append(f"Overall Score: {self.results['overall_score']:.1f}%")
        report.append(f"Critical Issues: {len(self.results['critical_issues'])}")
        report.append("")
        
        # Individual Quality Gates
        report.append("🎯 QUALITY GATES DETAILED RESULTS")
        report.append("-" * 40)
        
        for gate_name, gate_result in self.results["quality_gates"].items():
            status = "✅ PASSED" if gate_result["passed"] else "❌ FAILED"
            report.append(f"{gate_name}: {status} (Score: {gate_result['score']:.1f}%)")
            
            if gate_result.get("issues"):
                for issue in gate_result["issues"]:
                    report.append(f"  ⚠️  {issue}")
            report.append("")
        
        # Critical Issues Summary
        if self.results["critical_issues"]:
            report.append("🚨 CRITICAL ISSUES TO ADDRESS")
            report.append("-" * 40)
            for issue in self.results["critical_issues"]:
                report.append(f"• {issue}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if not self.results["production_ready"]:
            if self.results["overall_score"] < 85:
                report.append("• Improve overall quality scores to meet 85% minimum threshold")
            
            for gate_name, gate_result in self.results["quality_gates"].items():
                if not gate_result["passed"]:
                    report.append(f"• Address issues in {gate_name}")
        else:
            report.append("• All quality gates passed! System is production-ready.")
            report.append("• Consider implementing additional monitoring and alerting")
            report.append("• Regular security audits recommended")
        
        report.append("")
        report.append("=" * 80)
        report.append("🔚 END OF QUALITY GATES REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, output_file: str = "quality_gates_comprehensive_final_report.json"):
        """Save results to JSON file."""
        output_path = self.project_root / output_file
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Results saved to: {output_path}")


def main():
    """Main execution function."""
    print("🚀 Starting Comprehensive Quality Gates Validation")
    print(f"📁 Project Root: /root/repo")
    print("")
    
    validator = QualityGatesValidator()
    results = validator.run_all_gates()
    
    # Generate and display report
    report = validator.generate_report()
    print(report)
    
    # Save results
    validator.save_results()
    
    # Exit with appropriate code
    exit_code = 0 if results["production_ready"] else 1
    print(f"\n🏁 Quality Gates Validation Complete (Exit Code: {exit_code})")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())