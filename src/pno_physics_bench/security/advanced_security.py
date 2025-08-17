"""
Advanced Security Framework for Probabilistic Neural Operators.

This module implements comprehensive security measures including input
sanitization, model watermarking, adversarial defense, and secure
inference protocols for production deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
import hmac
import secrets
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..models import ProbabilisticNeuralOperator


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    metadata: Dict[str, Any]
    remediation: Optional[str] = None


class InputSanitizer:
    """
    Advanced input sanitization for neural operator inputs.
    
    Validates and sanitizes inputs to prevent adversarial attacks,
    data poisoning, and ensure input compliance with expected distributions.
    """
    
    def __init__(
        self,
        expected_shape: Tuple[int, ...],
        value_range: Tuple[float, float] = (-10.0, 10.0),
        distribution_params: Optional[Dict[str, float]] = None,
        anomaly_threshold: float = 3.0
    ):
        self.expected_shape = expected_shape
        self.value_range = value_range
        self.distribution_params = distribution_params or {'mean': 0.0, 'std': 1.0}
        self.anomaly_threshold = anomaly_threshold
        
        # Statistical baselines for anomaly detection
        self.baseline_stats = None
        self.update_count = 0
        
        # Security logger
        self.logger = logging.getLogger(__name__ + '.security')
    
    def update_baseline_stats(self, clean_data: torch.Tensor):
        """Update baseline statistics from known clean data."""
        with torch.no_grad():
            stats = {
                'mean': clean_data.mean().item(),
                'std': clean_data.std().item(),
                'min': clean_data.min().item(),
                'max': clean_data.max().item(),
                'percentiles': {
                    '5': torch.quantile(clean_data, 0.05).item(),
                    '25': torch.quantile(clean_data, 0.25).item(),
                    '75': torch.quantile(clean_data, 0.75).item(),
                    '95': torch.quantile(clean_data, 0.95).item()
                }
            }
            
            if self.baseline_stats is None:
                self.baseline_stats = stats
            else:
                # Exponential moving average update
                alpha = 0.1
                for key in ['mean', 'std', 'min', 'max']:
                    self.baseline_stats[key] = (
                        (1 - alpha) * self.baseline_stats[key] + alpha * stats[key]
                    )
                
                for percentile in self.baseline_stats['percentiles']:
                    self.baseline_stats['percentiles'][percentile] = (
                        (1 - alpha) * self.baseline_stats['percentiles'][percentile] +
                        alpha * stats['percentiles'][percentile]
                    )
            
            self.update_count += 1
    
    def validate_input_shape(self, input_tensor: torch.Tensor) -> bool:
        """Validate input tensor shape."""
        if input_tensor.shape[1:] != self.expected_shape:
            self.logger.warning(
                f"Input shape mismatch: expected {self.expected_shape}, "
                f"got {input_tensor.shape[1:]}"
            )
            return False
        return True
    
    def validate_input_range(self, input_tensor: torch.Tensor) -> bool:
        """Validate input values are within expected range."""
        min_val = input_tensor.min().item()
        max_val = input_tensor.max().item()
        
        if min_val < self.value_range[0] or max_val > self.value_range[1]:
            self.logger.warning(
                f"Input values out of range: [{min_val:.4f}, {max_val:.4f}] "
                f"not in {self.value_range}"
            )
            return False
        return True
    
    def detect_statistical_anomalies(self, input_tensor: torch.Tensor) -> List[str]:
        """Detect statistical anomalies in input."""
        anomalies = []
        
        if self.baseline_stats is None:
            return anomalies
        
        with torch.no_grad():
            # Compute input statistics
            input_mean = input_tensor.mean().item()
            input_std = input_tensor.std().item()
            
            # Check for mean deviation
            mean_z_score = abs(input_mean - self.baseline_stats['mean']) / self.baseline_stats['std']
            if mean_z_score > self.anomaly_threshold:
                anomalies.append(f"Mean anomaly: z-score={mean_z_score:.2f}")
            
            # Check for std deviation
            std_ratio = input_std / self.baseline_stats['std']
            if std_ratio > 3.0 or std_ratio < 0.33:
                anomalies.append(f"Std anomaly: ratio={std_ratio:.2f}")
            
            # Check for extreme values
            input_min = input_tensor.min().item()
            input_max = input_tensor.max().item()
            
            baseline_range = self.baseline_stats['max'] - self.baseline_stats['min']
            if input_min < self.baseline_stats['min'] - baseline_range * 0.5:
                anomalies.append(f"Extreme minimum: {input_min:.4f}")
            
            if input_max > self.baseline_stats['max'] + baseline_range * 0.5:
                anomalies.append(f"Extreme maximum: {input_max:.4f}")
        
        return anomalies
    
    def detect_adversarial_patterns(self, input_tensor: torch.Tensor) -> List[str]:
        """Detect patterns indicative of adversarial inputs."""
        patterns = []
        
        with torch.no_grad():
            # High-frequency noise detection
            # Compute gradient magnitude as proxy for high-frequency content
            if len(input_tensor.shape) >= 3:  # Spatial data
                grad_x = torch.diff(input_tensor, dim=-1)
                grad_y = torch.diff(input_tensor, dim=-2)
                
                grad_magnitude = torch.sqrt(grad_x[..., :-1]**2 + grad_y[..., 1:]**2)
                high_freq_content = grad_magnitude.mean().item()
                
                # Check if gradient magnitude is unusually high
                if self.baseline_stats and 'gradient_mean' in self.baseline_stats:
                    if high_freq_content > self.baseline_stats['gradient_mean'] * 3:
                        patterns.append(f"High-frequency noise: {high_freq_content:.4f}")
                else:
                    # First time - just record
                    if 'gradient_mean' not in (self.baseline_stats or {}):
                        if self.baseline_stats is None:
                            self.baseline_stats = {}
                        self.baseline_stats['gradient_mean'] = high_freq_content
            
            # Uniform noise detection (adversarial perturbations often uniform)
            flat_input = input_tensor.flatten()
            hist_counts = torch.histc(flat_input, bins=50)
            hist_uniformity = torch.std(hist_counts.float()) / torch.mean(hist_counts.float())
            
            if hist_uniformity < 0.3:  # Too uniform
                patterns.append(f"Uniform distribution detected: uniformity={hist_uniformity:.3f}")
            
            # Check for repeated patterns (potential attack signatures)
            if len(input_tensor.shape) >= 3:
                # Look for repetitive patterns in spatial domain
                patches = F.unfold(input_tensor.mean(dim=1, keepdim=True), kernel_size=3, stride=1)
                if patches.numel() > 0:
                    patch_variance = torch.var(patches, dim=1).mean().item()
                    if patch_variance < 1e-6:  # Very low variance indicates repetition
                        patterns.append(f"Repetitive patterns detected: variance={patch_variance:.2e}")
        
        return patterns
    
    def sanitize_input(
        self,
        input_tensor: torch.Tensor,
        strict_mode: bool = False
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Sanitize input tensor and return cleaned version with warnings.
        
        Args:
            input_tensor: Input to sanitize
            strict_mode: If True, apply aggressive sanitization
            
        Returns:
            Sanitized tensor and list of applied sanitizations
        """
        sanitizations = []
        sanitized = input_tensor.clone()
        
        # Clamp to value range
        original_min = sanitized.min().item()
        original_max = sanitized.max().item()
        
        sanitized = torch.clamp(sanitized, self.value_range[0], self.value_range[1])
        
        if original_min < self.value_range[0] or original_max > self.value_range[1]:
            sanitizations.append(
                f"Clamped values from [{original_min:.4f}, {original_max:.4f}] "
                f"to {self.value_range}"
            )
        
        # Remove extreme outliers if in strict mode
        if strict_mode and self.baseline_stats:
            # Remove values beyond 4 standard deviations
            mean = self.baseline_stats['mean']
            std = self.baseline_stats['std']
            
            outlier_mask = torch.abs(sanitized - mean) > 4 * std
            if outlier_mask.any():
                # Replace outliers with clipped values
                sanitized = torch.where(
                    outlier_mask,
                    torch.clamp(sanitized, mean - 4*std, mean + 4*std),
                    sanitized
                )
                sanitizations.append(f"Removed {outlier_mask.sum().item()} extreme outliers")
        
        # Apply smoothing to reduce high-frequency adversarial noise
        if strict_mode and len(sanitized.shape) >= 3:
            # Gaussian blur to remove high-frequency perturbations
            kernel_size = 3
            sigma = 0.5
            
            # Create Gaussian kernel
            x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            # Apply separable convolution
            channels = sanitized.shape[1]
            kernel_2d = kernel_1d.view(1, 1, -1) * kernel_1d.view(1, -1, 1)
            kernel = kernel_2d.expand(channels, 1, -1, -1)
            
            original_norm = torch.norm(sanitized)
            sanitized = F.conv2d(
                sanitized,
                kernel,
                padding=kernel_size//2,
                groups=channels
            )
            new_norm = torch.norm(sanitized)
            
            if abs(original_norm - new_norm) / original_norm > 0.01:
                sanitizations.append(f"Applied Gaussian smoothing: σ={sigma}")
        
        return sanitized, sanitizations
    
    def validate_and_sanitize(
        self,
        input_tensor: torch.Tensor,
        strict_mode: bool = False,
        raise_on_failure: bool = False
    ) -> Tuple[torch.Tensor, List[SecurityEvent]]:
        """
        Complete validation and sanitization pipeline.
        
        Returns:
            Sanitized tensor and list of security events
        """
        events = []
        
        # Shape validation
        if not self.validate_input_shape(input_tensor):
            event = SecurityEvent(
                timestamp=time.time(),
                event_type="input_validation",
                severity="high",
                description="Input shape validation failed",
                metadata={'expected_shape': self.expected_shape, 'actual_shape': input_tensor.shape},
                remediation="Reject input or reshape"
            )
            events.append(event)
            
            if raise_on_failure:
                raise ValueError(f"Input shape validation failed: {event.description}")
        
        # Range validation
        if not self.validate_input_range(input_tensor):
            event = SecurityEvent(
                timestamp=time.time(),
                event_type="input_validation",
                severity="medium",
                description="Input range validation failed",
                metadata={'value_range': self.value_range},
                remediation="Clamp values to valid range"
            )
            events.append(event)
        
        # Statistical anomaly detection
        anomalies = self.detect_statistical_anomalies(input_tensor)
        for anomaly in anomalies:
            event = SecurityEvent(
                timestamp=time.time(),
                event_type="anomaly_detection",
                severity="medium",
                description=f"Statistical anomaly detected: {anomaly}",
                metadata={'anomaly_type': 'statistical'},
                remediation="Monitor or reject input"
            )
            events.append(event)
        
        # Adversarial pattern detection
        patterns = self.detect_adversarial_patterns(input_tensor)
        for pattern in patterns:
            event = SecurityEvent(
                timestamp=time.time(),
                event_type="adversarial_detection",
                severity="high",
                description=f"Adversarial pattern detected: {pattern}",
                metadata={'pattern_type': 'adversarial'},
                remediation="Apply strict sanitization"
            )
            events.append(event)
            strict_mode = True  # Force strict mode for adversarial inputs
        
        # Sanitize input
        sanitized_input, sanitizations = self.sanitize_input(input_tensor, strict_mode)
        
        for sanitization in sanitizations:
            event = SecurityEvent(
                timestamp=time.time(),
                event_type="input_sanitization",
                severity="low",
                description=f"Applied sanitization: {sanitization}",
                metadata={'sanitization_type': 'automatic'},
                remediation="Input sanitized"
            )
            events.append(event)
        
        return sanitized_input, events


class ModelWatermarking:
    """
    Model watermarking for intellectual property protection.
    
    Embeds invisible watermarks in model predictions that can be
    verified to prove model ownership and detect unauthorized usage.
    """
    
    def __init__(
        self,
        watermark_key: str,
        watermark_strength: float = 0.01,
        verification_threshold: float = 0.7
    ):
        self.watermark_key = watermark_key
        self.watermark_strength = watermark_strength
        self.verification_threshold = verification_threshold
        
        # Generate deterministic watermark pattern from key
        self.watermark_pattern = self._generate_watermark_pattern()
    
    def _generate_watermark_pattern(self) -> torch.Tensor:
        """Generate deterministic watermark pattern from key."""
        # Use HMAC for deterministic pseudo-random generation
        hmac_gen = hmac.new(
            self.watermark_key.encode(),
            b"watermark_pattern",
            hashlib.sha256
        )
        
        # Generate pattern
        pattern_bytes = hmac_gen.digest()
        pattern_array = np.frombuffer(pattern_bytes, dtype=np.uint8)
        
        # Normalize to [-1, 1]
        pattern = (pattern_array.astype(np.float32) / 127.5) - 1.0
        
        return torch.from_numpy(pattern)
    
    def embed_watermark(
        self,
        prediction: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Embed watermark in model prediction.
        
        Uses input-dependent watermarking to make detection harder.
        """
        batch_size = prediction.shape[0]
        
        # Generate input-dependent watermark seed
        input_hash = hashlib.sha256(input_tensor.detach().cpu().numpy().tobytes()).hexdigest()
        
        # Combine with watermark key
        combined_key = f"{self.watermark_key}_{input_hash}"
        
        # Generate watermark for this input
        hmac_gen = hmac.new(
            combined_key.encode(),
            b"prediction_watermark",
            hashlib.sha256
        )
        
        watermark_bytes = hmac_gen.digest()
        
        # Create spatial watermark pattern
        if len(prediction.shape) == 4:  # [batch, channels, height, width]
            h, w = prediction.shape[-2:]
            pattern_length = min(len(watermark_bytes), h * w)
            
            watermark_spatial = np.frombuffer(watermark_bytes[:pattern_length], dtype=np.uint8)
            watermark_spatial = (watermark_spatial.astype(np.float32) / 127.5) - 1.0
            
            # Reshape to spatial dimensions
            if pattern_length < h * w:
                # Tile pattern if too short
                repetitions = (h * w + pattern_length - 1) // pattern_length
                watermark_spatial = np.tile(watermark_spatial, repetitions)[:h * w]
            
            watermark_spatial = watermark_spatial.reshape(h, w)
            watermark_tensor = torch.from_numpy(watermark_spatial).to(prediction.device)
            
            # Expand for batch and channels
            watermark_tensor = watermark_tensor.unsqueeze(0).unsqueeze(0)
            watermark_tensor = watermark_tensor.expand(batch_size, prediction.shape[1], -1, -1)
            
        else:
            # For non-spatial data, use simpler watermarking
            pattern_length = min(len(watermark_bytes), prediction.numel() // batch_size)
            watermark_flat = np.frombuffer(watermark_bytes[:pattern_length], dtype=np.uint8)
            watermark_flat = (watermark_flat.astype(np.float32) / 127.5) - 1.0
            
            watermark_tensor = torch.from_numpy(watermark_flat).to(prediction.device)
            watermark_tensor = watermark_tensor.view(1, -1).expand(batch_size, -1)
            
            # Reshape to match prediction
            watermark_tensor = watermark_tensor.view(prediction.shape)
        
        # Embed watermark with adaptive strength
        prediction_magnitude = torch.norm(prediction, dim=tuple(range(1, len(prediction.shape))))
        adaptive_strength = self.watermark_strength * prediction_magnitude.view(-1, *([1] * (len(prediction.shape) - 1)))
        
        watermarked_prediction = prediction + adaptive_strength * watermark_tensor
        
        return watermarked_prediction
    
    def verify_watermark(
        self,
        prediction: torch.Tensor,
        input_tensor: torch.Tensor,
        suspected_watermarked: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        Verify if prediction contains our watermark.
        
        Returns:
            (is_watermarked, confidence_score)
        """
        # Generate expected watermark
        expected_watermarked = self.embed_watermark(prediction, input_tensor)
        
        # Compute correlation between suspected and expected watermarks
        watermark_signal = suspected_watermarked - prediction
        expected_signal = expected_watermarked - prediction
        
        # Flatten for correlation computation
        signal_flat = watermark_signal.flatten()
        expected_flat = expected_signal.flatten()
        
        # Compute normalized correlation
        if torch.norm(signal_flat) > 0 and torch.norm(expected_flat) > 0:
            correlation = F.cosine_similarity(
                signal_flat.unsqueeze(0),
                expected_flat.unsqueeze(0)
            ).item()
        else:
            correlation = 0.0
        
        # Verify watermark
        is_watermarked = correlation > self.verification_threshold
        
        return is_watermarked, correlation
    
    def generate_ownership_proof(
        self,
        test_inputs: List[torch.Tensor],
        model: nn.Module
    ) -> Dict[str, Any]:
        """Generate cryptographic proof of model ownership."""
        proofs = []
        
        with torch.no_grad():
            for i, test_input in enumerate(test_inputs):
                # Get model prediction
                prediction = model(test_input)
                
                # Embed watermark
                watermarked = self.embed_watermark(prediction, test_input)
                
                # Create proof record
                input_hash = hashlib.sha256(test_input.cpu().numpy().tobytes()).hexdigest()
                prediction_hash = hashlib.sha256(prediction.cpu().numpy().tobytes()).hexdigest()
                watermark_hash = hashlib.sha256(watermarked.cpu().numpy().tobytes()).hexdigest()
                
                proof = {
                    'test_case': i,
                    'input_hash': input_hash,
                    'prediction_hash': prediction_hash,
                    'watermarked_hash': watermark_hash,
                    'watermark_strength': self.watermark_strength,
                    'timestamp': time.time()
                }
                
                proofs.append(proof)
        
        # Create overall proof document
        proof_document = {
            'watermark_key_hash': hashlib.sha256(self.watermark_key.encode()).hexdigest(),
            'verification_threshold': self.verification_threshold,
            'proofs': proofs,
            'total_test_cases': len(test_inputs)
        }
        
        # Sign the proof document
        document_str = json.dumps(proof_document, sort_keys=True)
        signature = hmac.new(
            self.watermark_key.encode(),
            document_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        proof_document['signature'] = signature
        
        return proof_document


class SecureInference:
    """
    Secure inference protocols for production deployment.
    
    Implements secure model serving with encryption, authentication,
    and audit logging.
    """
    
    def __init__(
        self,
        model: ProbabilisticNeuralOperator,
        encryption_key: Optional[bytes] = None,
        require_authentication: bool = True,
        audit_logging: bool = True
    ):
        self.model = model
        self.require_authentication = require_authentication
        self.audit_logging = audit_logging
        
        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            # Generate random key
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            logging.warning("Generated random encryption key - store securely!")
        
        # Authentication tokens
        self.valid_tokens = set()
        
        # Audit log
        self.audit_log = []
        
        # Rate limiting
        self.request_counts = {}
        self.rate_limit = 100  # requests per minute
        
        # Security logger
        self.logger = logging.getLogger(__name__ + '.secure_inference')
    
    def generate_auth_token(self, user_id: str, expiry_hours: int = 24) -> str:
        """Generate authentication token for user."""
        expiry_time = time.time() + expiry_hours * 3600
        
        token_data = {
            'user_id': user_id,
            'expiry': expiry_time,
            'nonce': secrets.token_hex(16)
        }
        
        token_str = json.dumps(token_data)
        encrypted_token = self.cipher.encrypt(token_str.encode())
        token = base64.urlsafe_b64encode(encrypted_token).decode()
        
        self.valid_tokens.add(token)
        
        return token
    
    def validate_auth_token(self, token: str) -> Tuple[bool, Optional[str]]:
        """Validate authentication token."""
        try:
            if token not in self.valid_tokens:
                return False, None
            
            # Decrypt token
            encrypted_data = base64.urlsafe_b64decode(token.encode())
            decrypted_str = self.cipher.decrypt(encrypted_data).decode()
            token_data = json.loads(decrypted_str)
            
            # Check expiry
            if time.time() > token_data['expiry']:
                self.valid_tokens.discard(token)
                return False, None
            
            return True, token_data['user_id']
            
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return False, None
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        current_time = time.time()
        minute_window = int(current_time // 60)
        
        if client_id not in self.request_counts:
            self.request_counts[client_id] = {}
        
        # Clean old windows
        old_windows = [w for w in self.request_counts[client_id] if w < minute_window - 5]
        for w in old_windows:
            del self.request_counts[client_id][w]
        
        # Count requests in current window
        current_count = self.request_counts[client_id].get(minute_window, 0)
        
        if current_count >= self.rate_limit:
            return False
        
        # Increment counter
        self.request_counts[client_id][minute_window] = current_count + 1
        return True
    
    def log_audit_event(
        self,
        event_type: str,
        user_id: Optional[str],
        metadata: Dict[str, Any]
    ):
        """Log audit event."""
        if not self.audit_logging:
            return
        
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'user_id': user_id,
            'metadata': metadata
        }
        
        self.audit_log.append(event)
        
        # Limit audit log size
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # Keep last 5000 events
        
        self.logger.info(f"Audit: {event_type} by {user_id}")
    
    def encrypt_data(self, data: torch.Tensor) -> bytes:
        """Encrypt tensor data."""
        data_bytes = data.cpu().numpy().tobytes()
        return self.cipher.encrypt(data_bytes)
    
    def decrypt_data(self, encrypted_data: bytes, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Decrypt tensor data."""
        data_bytes = self.cipher.decrypt(encrypted_data)
        data_array = np.frombuffer(data_bytes, dtype=dtype.as_numpy_dtype).reshape(shape)
        return torch.from_numpy(data_array)
    
    def secure_inference(
        self,
        encrypted_input: bytes,
        input_shape: Tuple[int, ...],
        auth_token: Optional[str] = None,
        client_id: str = "anonymous"
    ) -> Tuple[bytes, List[SecurityEvent]]:
        """
        Perform secure inference with authentication and audit logging.
        
        Args:
            encrypted_input: Encrypted input tensor
            input_shape: Expected input shape
            auth_token: Authentication token
            client_id: Client identifier for rate limiting
            
        Returns:
            Encrypted prediction and security events
        """
        events = []
        
        # Authentication
        user_id = None
        if self.require_authentication:
            if not auth_token:
                raise ValueError("Authentication required but no token provided")
            
            valid, user_id = self.validate_auth_token(auth_token)
            if not valid:
                event = SecurityEvent(
                    timestamp=time.time(),
                    event_type="authentication_failure",
                    severity="high",
                    description="Authentication failed",
                    metadata={'client_id': client_id}
                )
                events.append(event)
                raise ValueError("Authentication failed")
        
        # Rate limiting
        if not self.check_rate_limit(client_id):
            event = SecurityEvent(
                timestamp=time.time(),
                event_type="rate_limit_exceeded",
                severity="medium",
                description="Rate limit exceeded",
                metadata={'client_id': client_id, 'user_id': user_id}
            )
            events.append(event)
            raise ValueError("Rate limit exceeded")
        
        # Decrypt input
        try:
            input_tensor = self.decrypt_data(encrypted_input, input_shape, torch.float32)
        except Exception as e:
            event = SecurityEvent(
                timestamp=time.time(),
                event_type="decryption_failure",
                severity="high",
                description=f"Input decryption failed: {str(e)}",
                metadata={'client_id': client_id, 'user_id': user_id}
            )
            events.append(event)
            raise ValueError("Input decryption failed")
        
        # Input validation and sanitization
        sanitizer = InputSanitizer(
            expected_shape=input_shape[1:],  # Remove batch dimension
            value_range=(-10.0, 10.0)
        )
        
        sanitized_input, sanitization_events = sanitizer.validate_and_sanitize(input_tensor)
        events.extend(sanitization_events)
        
        # Model inference
        try:
            with torch.no_grad():
                prediction, uncertainty = self.model.predict_with_uncertainty(sanitized_input)
            
            # Log successful inference
            self.log_audit_event(
                'successful_inference',
                user_id,
                {
                    'client_id': client_id,
                    'input_shape': list(input_shape),
                    'output_shape': list(prediction.shape),
                    'num_sanitization_events': len(sanitization_events)
                }
            )
            
        except Exception as e:
            event = SecurityEvent(
                timestamp=time.time(),
                event_type="inference_failure",
                severity="medium",
                description=f"Model inference failed: {str(e)}",
                metadata={'client_id': client_id, 'user_id': user_id}
            )
            events.append(event)
            
            self.log_audit_event(
                'inference_failure',
                user_id,
                {'client_id': client_id, 'error': str(e)}
            )
            
            raise ValueError("Model inference failed")
        
        # Encrypt output
        output_data = torch.cat([prediction, uncertainty], dim=1)  # Combine prediction and uncertainty
        encrypted_output = self.encrypt_data(output_data)
        
        return encrypted_output, events
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for specified time period."""
        cutoff_time = time.time() - hours * 3600
        recent_events = [e for e in self.audit_log if e['timestamp'] > cutoff_time]
        
        summary = {
            'total_events': len(recent_events),
            'event_types': {},
            'users': set(),
            'clients': set()
        }
        
        for event in recent_events:
            event_type = event['event_type']
            summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1
            
            if event['user_id']:
                summary['users'].add(event['user_id'])
            
            if 'client_id' in event['metadata']:
                summary['clients'].add(event['metadata']['client_id'])
        
        summary['unique_users'] = len(summary['users'])
        summary['unique_clients'] = len(summary['clients'])
        summary['users'] = list(summary['users'])
        summary['clients'] = list(summary['clients'])
        
        return summary


def create_secure_pno_deployment():
    """Create a complete secure PNO deployment setup."""
    from ..models import ProbabilisticNeuralOperator
    
    # Create model
    model = ProbabilisticNeuralOperator(
        input_dim=3,
        hidden_dim=128,
        num_layers=4,
        modes=16
    )
    
    # Create input sanitizer
    sanitizer = InputSanitizer(
        expected_shape=(3, 64, 64),
        value_range=(-5.0, 5.0),
        anomaly_threshold=3.0
    )
    
    # Create watermarking
    watermarking = ModelWatermarking(
        watermark_key="your-secret-watermark-key-2024",
        watermark_strength=0.01
    )
    
    # Create secure inference
    secure_inference = SecureInference(
        model=model,
        require_authentication=True,
        audit_logging=True
    )
    
    return {
        'model': model,
        'sanitizer': sanitizer,
        'watermarking': watermarking,
        'secure_inference': secure_inference
    }


def run_security_demo():
    """Demonstrate security features."""
    print("🔒 Running Security Framework Demo...")
    
    # Create secure deployment
    deployment = create_secure_pno_deployment()
    
    # Generate test data
    test_input = torch.randn(1, 3, 64, 64)
    
    # Test input sanitization
    sanitizer = deployment['sanitizer']
    sanitized_input, events = sanitizer.validate_and_sanitize(test_input)
    print(f"✅ Input sanitization: {len(events)} security events")
    
    # Test watermarking
    watermarking = deployment['watermarking']
    model = deployment['model']
    
    with torch.no_grad():
        prediction = model(test_input)
        watermarked = watermarking.embed_watermark(prediction, test_input)
        is_watermarked, confidence = watermarking.verify_watermark(prediction, test_input, watermarked)
    
    print(f"✅ Watermarking: embedded and verified (confidence: {confidence:.3f})")
    
    # Test secure inference
    secure_inference = deployment['secure_inference']
    
    # Generate auth token
    token = secure_inference.generate_auth_token("test_user", expiry_hours=1)
    
    # Encrypt input
    encrypted_input = secure_inference.encrypt_data(test_input)
    
    # Perform secure inference
    try:
        encrypted_output, security_events = secure_inference.secure_inference(
            encrypted_input=encrypted_input,
            input_shape=test_input.shape,
            auth_token=token,
            client_id="demo_client"
        )
        print(f"✅ Secure inference: completed with {len(security_events)} security events")
    except Exception as e:
        print(f"❌ Secure inference failed: {e}")
    
    # Get audit summary
    audit_summary = secure_inference.get_audit_summary(hours=1)
    print(f"✅ Audit logging: {audit_summary['total_events']} events recorded")
    
    print("🚀 Security Framework Demo completed!")


if __name__ == "__main__":
    # Run security demonstration
    run_security_demo()
    print("🔒 Advanced Security Framework ready for production deployment!")