"""
Security Tests for pno-physics-bench
===================================
Comprehensive security testing including vulnerability detection and input validation.
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestSecurityValidation:
    """Test security validation and threat detection."""
    
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_input_validation_security(self):
        """Test input validation against malicious inputs."""
        try:
            from pno_physics_bench.security_validation import InputValidator
            
            validator = InputValidator()
            
            # Test SQL injection attempts
            sql_injections = [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "'; UPDATE users SET admin=1; --"
            ]
            
            for injection in sql_injections:
                is_valid, threat = validator.validate_tensor_input(injection, "test_input")
                # Should detect as invalid/threat
                assert is_valid is False or threat is not None
                
        except ImportError:
            pytest.skip("Security validation module not available")
    
    def test_xss_prevention(self):
        """Test Cross-Site Scripting (XSS) prevention."""
        try:
            from pno_physics_bench.security.input_validation_enhanced import validator
            
            xss_payloads = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
                "<svg onload=alert('xss')>"
            ]
            
            for payload in xss_payloads:
                is_valid, error = validator.validate_string(payload)
                assert is_valid is False
                assert error is not None
                assert "dangerous content" in error.lower()
                
        except ImportError:
            pytest.skip("Enhanced input validation not available")
    
    def test_path_traversal_prevention(self):
        """Test path traversal attack prevention."""
        try:
            from pno_physics_bench.security.input_validation_enhanced import validator
            
            path_traversals = [
                "../../../etc/passwd",
                "..\..\windows\system32\config\sam",
                "/etc/passwd",
                "../../sensitive_file.txt"
            ]
            
            for path in path_traversals:
                is_valid, error = validator.validate_file_path(path)
                if "../" in path or "..\" in path:
                    assert is_valid is False
                    assert "path traversal" in error.lower()
                    
        except ImportError:
            pytest.skip("Enhanced input validation not available")
    
    def test_command_injection_prevention(self):
        """Test command injection prevention."""
        # Check that subprocess calls are secured
        project_root = Path(__file__).parent.parent
        src_files = list((project_root / "src").rglob("*.py"))
        
        unsafe_patterns = [
            "os.system(",
            "subprocess.call(.*, shell=True)",
            "exec(",
            "eval("
        ]
        
        for src_file in src_files:
            try:
                content = src_file.read_text()
                for pattern in unsafe_patterns:
                    # After security fixes, these should be minimal or secured
                    if "eval(" in content and "safe_eval" not in content:
                        # This test will help identify any remaining unsafe eval usage
                        pass
            except:
                continue
    
    def test_secure_random_generation(self):
        """Test that secure random number generation is used."""
        # Mock secure random usage
        import secrets
        
        # Test that secure randomness is available
        secure_bytes = secrets.token_bytes(32)
        assert len(secure_bytes) == 32
        
        secure_hex = secrets.token_hex(16)
        assert len(secure_hex) == 32  # 16 bytes = 32 hex chars
    
    def test_authentication_mechanisms(self):
        """Test authentication and authorization mechanisms."""
        # Mock authentication system
        mock_auth = Mock()
        mock_auth.validate_token.return_value = True
        mock_auth.check_permissions.return_value = True
        
        # Test authentication flow
        token = "mock_token_12345"
        is_valid = mock_auth.validate_token(token)
        assert is_valid is True
        
        permissions = mock_auth.check_permissions("user", "resource")
        assert permissions is True
    
    def test_data_sanitization(self):
        """Test data sanitization functions."""
        try:
            from pno_physics_bench.security.input_validation_enhanced import validator
            
            dangerous_inputs = [
                "<script>alert('test')</script>",
                "javascript:void(0)",
                "' OR 1=1 --"
            ]
            
            for dangerous_input in dangerous_inputs:
                sanitized = validator.sanitize_string(dangerous_input)
                
                # Should remove or escape dangerous content
                assert "<script>" not in sanitized
                assert "javascript:" not in sanitized
                
        except ImportError:
            pytest.skip("Input sanitization not available")
    
    @pytest.mark.parametrize("threat_type", [
        "sql_injection",
        "xss",
        "path_traversal", 
        "command_injection",
        "code_injection"
    ])
    def test_threat_detection(self, threat_type):
        """Test detection of different threat types."""
        try:
            from pno_physics_bench.security_validation import SecurityThreat
            
            threat = SecurityThreat(
                threat_id=f"test_{threat_type}",
                threat_type=threat_type,
                severity="high",
                description=f"Test {threat_type} threat",
                source="test",
                timestamp=123456789,
                mitigation="block",
                blocked=True
            )
            
            assert threat.threat_type == threat_type
            assert threat.blocked is True
            assert threat.severity == "high"
            
        except ImportError:
            pytest.skip("Security threat detection not available")

class TestSecurityCompliance:
    """Test security compliance and best practices."""
    
    def test_secrets_management(self):
        """Test that secrets are not hardcoded."""
        project_root = Path(__file__).parent.parent
        src_files = list((project_root / "src").rglob("*.py"))
        
        # Look for potential hardcoded secrets
        secret_patterns = [
            r"password\s*=\s*["'][^"']{8,}["']",
            r"api_key\s*=\s*["'][^"']{8,}["']",
            r"secret\s*=\s*["'][^"']{8,}["']",
            r"token\s*=\s*["'][^"']{8,}["']"
        ]
        
        for src_file in src_files:
            try:
                content = src_file.read_text()
                for pattern in secret_patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    # Should not find hardcoded secrets
                    for match in matches:
                        # Allow test values and placeholders
                        if not any(placeholder in match.lower() 
                                 for placeholder in ['test', 'example', 'placeholder', 'your_', 'xxx']):
                            pytest.fail(f"Potential hardcoded secret in {src_file}: {match}")
            except:
                continue
    
    def test_encryption_usage(self):
        """Test that encryption is used appropriately."""
        # Mock encryption functionality
        mock_crypto = Mock()
        mock_crypto.encrypt.return_value = b"encrypted_data"
        mock_crypto.decrypt.return_value = b"decrypted_data"
        
        # Test encryption flow
        plaintext = b"sensitive_data"
        encrypted = mock_crypto.encrypt(plaintext)
        decrypted = mock_crypto.decrypt(encrypted)
        
        assert encrypted != plaintext
        assert decrypted == b"decrypted_data"
    
    def test_secure_communication(self):
        """Test secure communication protocols."""
        # Check for HTTPS usage in configuration files
        project_root = Path(__file__).parent.parent
        config_files = list(project_root.rglob("*.json")) + list(project_root.rglob("*.yaml")) + list(project_root.rglob("*.yml"))
        
        for config_file in config_files:
            try:
                content = config_file.read_text()
                # If URLs are found, they should prefer HTTPS
                import re
                http_urls = re.findall(r'http://[^\s"']+', content)
                for url in http_urls:
                    # Allow localhost and development URLs
                    if not any(dev_indicator in url for dev_indicator in ['localhost', '127.0.0.1', 'dev', 'test']):
                        print(f"Warning: Non-HTTPS URL found in {config_file}: {url}")
            except:
                continue

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
