"""Adaptive Security Framework for PNO Systems.

This module implements advanced security mechanisms specifically designed for
adaptive neural operators, including adversarial detection, input sanitization,
and model integrity verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import hmac
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass
import numpy as np
from collections import deque
import threading
import secrets
import warnings


class ThreatLevel(Enum):
    """Security threat levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of attacks on neural operators."""
    ADVERSARIAL_INPUT = "adversarial_input"
    MODEL_POISONING = "model_poisoning"
    GRADIENT_ATTACK = "gradient_attack"
    UNCERTAINTY_MANIPULATION = "uncertainty_manipulation"
    DATA_EXTRACTION = "data_extraction"
    BACKDOOR_INJECTION = "backdoor_injection"
    EVASION_ATTACK = "evasion_attack"
    INFERENCE_ATTACK = "inference_attack"


@dataclass
class SecurityIncident:
    """Record of a security incident."""
    timestamp: float
    attack_type: AttackType
    threat_level: ThreatLevel
    confidence_score: float
    input_hash: str
    attack_vector: Dict[str, Any]
    mitigation_applied: str
    success: bool


class AdversarialDetector:
    """Advanced adversarial input detection for neural operators."""
    
    def __init__(self, sensitivity_threshold: float = 0.1, window_size: int = 100):
        self.sensitivity_threshold = sensitivity_threshold
        self.window_size = window_size
        
        # Detection history
        self.detection_history = deque(maxlen=window_size)
        self.baseline_statistics = {}
        self.adaptive_threshold = sensitivity_threshold
        
        # Statistical anomaly detection
        self.input_statistics = {
            'mean': 0.0,
            'std': 1.0,
            'min': float('inf'),
            'max': float('-inf'),
            'gradient_stats': {'mean': 0.0, 'std': 1.0}
        }
        
        logging.info(f"Initialized Adversarial Detector with sensitivity {sensitivity_threshold}")
    
    def detect_adversarial_input(self, input_tensor: torch.Tensor, model: nn.Module) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect adversarial inputs using multiple detection methods.
        
        Returns:
            (is_adversarial, confidence_score, detection_details)
        """
        detection_results = {}
        
        # 1. Statistical anomaly detection
        stats_score = self._detect_statistical_anomaly(input_tensor)
        detection_results['statistical_anomaly'] = stats_score
        
        # 2. Gradient-based detection
        gradient_score = self._detect_gradient_anomaly(input_tensor, model)
        detection_results['gradient_anomaly'] = gradient_score
        
        # 3. Uncertainty-based detection
        uncertainty_score = self._detect_uncertainty_anomaly(input_tensor, model)
        detection_results['uncertainty_anomaly'] = uncertainty_score
        
        # 4. Input reconstruction detection
        reconstruction_score = self._detect_reconstruction_anomaly(input_tensor)
        detection_results['reconstruction_anomaly'] = reconstruction_score
        
        # 5. Frequency domain analysis
        frequency_score = self._detect_frequency_anomaly(input_tensor)
        detection_results['frequency_anomaly'] = frequency_score
        
        # Combine scores using weighted average
        weights = {
            'statistical_anomaly': 0.2,
            'gradient_anomaly': 0.3,
            'uncertainty_anomaly': 0.25,
            'reconstruction_anomaly': 0.15,
            'frequency_anomaly': 0.1
        }
        
        combined_score = sum(score * weights.get(method, 0.2) 
                           for method, score in detection_results.items())
        
        # Adaptive threshold based on recent detections
        is_adversarial = combined_score > self.adaptive_threshold
        
        # Update detection history and adaptive threshold
        self._update_detection_history(combined_score, is_adversarial)
        
        detection_details = {
            'individual_scores': detection_results,
            'combined_score': combined_score,
            'threshold_used': self.adaptive_threshold,
            'detection_methods': list(detection_results.keys())
        }
        
        return is_adversarial, combined_score, detection_details
    
    def _detect_statistical_anomaly(self, input_tensor: torch.Tensor) -> float:
        """Detect statistical anomalies in input."""
        try:
            # Flatten input for statistical analysis
            flattened = input_tensor.flatten()
            
            current_mean = float(torch.mean(flattened))
            current_std = float(torch.std(flattened))
            current_min = float(torch.min(flattened))
            current_max = float(torch.max(flattened))
            
            # Compare with baseline statistics
            if self.baseline_statistics:
                mean_diff = abs(current_mean - self.baseline_statistics.get('mean', current_mean))
                std_diff = abs(current_std - self.baseline_statistics.get('std', current_std))
                range_diff = abs((current_max - current_min) - 
                               (self.baseline_statistics.get('max', current_max) - 
                                self.baseline_statistics.get('min', current_min)))
                
                # Normalized anomaly score
                anomaly_score = (mean_diff + std_diff + range_diff) / 3.0
                return min(1.0, anomaly_score)
            else:
                # Initialize baseline
                self.baseline_statistics = {
                    'mean': current_mean,
                    'std': current_std,
                    'min': current_min,
                    'max': current_max
                }
                return 0.0
                
        except Exception as e:
            logging.warning(f"Statistical anomaly detection failed: {e}")
            return 0.0
    
    def _detect_gradient_anomaly(self, input_tensor: torch.Tensor, model: nn.Module) -> float:
        """Detect gradient-based attacks."""
        try:
            input_tensor.requires_grad_(True)
            
            # Forward pass
            output = model(input_tensor.unsqueeze(0) if len(input_tensor.shape) == 3 else input_tensor)
            
            # Compute gradient with respect to input
            grad_output = torch.ones_like(output)
            input_gradient = torch.autograd.grad(
                outputs=output, 
                inputs=input_tensor,
                grad_outputs=grad_output,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            
            # Gradient magnitude analysis
            grad_magnitude = torch.norm(input_gradient).item()
            grad_variance = torch.var(input_gradient).item()
            
            # Compare with expected gradient patterns
            if 'gradient_stats' in self.input_statistics:
                expected_magnitude = self.input_statistics['gradient_stats']['mean']
                expected_variance = self.input_statistics['gradient_stats']['std']
                
                magnitude_anomaly = abs(grad_magnitude - expected_magnitude) / (expected_magnitude + 1e-8)
                variance_anomaly = abs(grad_variance - expected_variance) / (expected_variance + 1e-8)
                
                return min(1.0, (magnitude_anomaly + variance_anomaly) / 2.0)
            else:
                # Initialize gradient statistics
                self.input_statistics['gradient_stats'] = {
                    'mean': grad_magnitude,
                    'std': grad_variance
                }
                return 0.0
                
        except Exception as e:
            logging.warning(f"Gradient anomaly detection failed: {e}")
            return 0.0
        finally:
            input_tensor.requires_grad_(False)
    
    def _detect_uncertainty_anomaly(self, input_tensor: torch.Tensor, model: nn.Module) -> float:
        """Detect anomalies in uncertainty estimates."""
        try:
            if hasattr(model, 'predict_with_uncertainty'):
                with torch.no_grad():
                    mean, uncertainty = model.predict_with_uncertainty(
                        input_tensor.unsqueeze(0) if len(input_tensor.shape) == 3 else input_tensor,
                        num_samples=20
                    )
                
                # Analyze uncertainty patterns
                avg_uncertainty = torch.mean(uncertainty).item()
                uncertainty_variance = torch.var(uncertainty).item()
                
                # Adversarial inputs often produce unusual uncertainty patterns
                # Very low or very high uncertainty can be suspicious
                if avg_uncertainty < 0.001 or avg_uncertainty > 1.0:
                    return min(1.0, abs(avg_uncertainty - 0.1) * 10.0)
                
                return 0.0
            else:
                return 0.0
                
        except Exception as e:
            logging.warning(f"Uncertainty anomaly detection failed: {e}")
            return 0.0
    
    def _detect_reconstruction_anomaly(self, input_tensor: torch.Tensor) -> float:
        """Detect anomalies using autoencoder-based reconstruction."""
        try:
            # Simple reconstruction check using SVD
            # In practice, this would use a trained autoencoder
            
            # Reshape for matrix operations
            original_shape = input_tensor.shape
            reshaped = input_tensor.view(-1, original_shape[-1])
            
            # SVD-based low-rank approximation
            U, S, V = torch.svd(reshaped)
            
            # Reconstruct using top-k singular values
            k = min(10, len(S))  # Use top 10 components
            reconstructed = torch.mm(torch.mm(U[:, :k], torch.diag(S[:k])), V[:, :k].t())
            reconstructed = reconstructed.view(original_shape)
            
            # Reconstruction error
            reconstruction_error = torch.norm(input_tensor - reconstructed).item()
            
            # Normalize error (simple normalization)
            normalized_error = reconstruction_error / (torch.norm(input_tensor).item() + 1e-8)
            
            return min(1.0, normalized_error)
            
        except Exception as e:
            logging.warning(f"Reconstruction anomaly detection failed: {e}")
            return 0.0
    
    def _detect_frequency_anomaly(self, input_tensor: torch.Tensor) -> float:
        """Detect anomalies in frequency domain."""
        try:
            # Apply FFT to detect frequency domain anomalies
            if len(input_tensor.shape) >= 2:
                # 2D FFT for spatial data
                fft_result = torch.fft.fft2(input_tensor[-2:] if len(input_tensor.shape) > 2 
                                          else input_tensor)
                
                # Analyze frequency spectrum
                magnitude_spectrum = torch.abs(fft_result)
                
                # Check for unusual frequency patterns
                # High-frequency components often indicate adversarial noise
                high_freq_energy = torch.sum(magnitude_spectrum[magnitude_spectrum.shape[0]//2:, 
                                                                magnitude_spectrum.shape[1]//2:])
                total_energy = torch.sum(magnitude_spectrum)
                
                high_freq_ratio = (high_freq_energy / (total_energy + 1e-8)).item()
                
                # Suspicious if high frequency ratio is unusual
                return min(1.0, max(0.0, (high_freq_ratio - 0.1) * 10.0))
            else:
                return 0.0
                
        except Exception as e:
            logging.warning(f"Frequency anomaly detection failed: {e}")
            return 0.0
    
    def _update_detection_history(self, score: float, is_adversarial: bool):
        """Update detection history and adaptive threshold."""
        self.detection_history.append({
            'score': score,
            'is_adversarial': is_adversarial,
            'timestamp': time.time()
        })
        
        # Adaptive threshold adjustment
        if len(self.detection_history) >= 10:
            recent_scores = [d['score'] for d in list(self.detection_history)[-10:]]
            recent_detections = [d['is_adversarial'] for d in list(self.detection_history)[-10:]]
            
            # Adjust threshold based on false positive/negative rates
            false_positive_rate = sum(1 for i, detected in enumerate(recent_detections) 
                                    if detected and recent_scores[i] < self.sensitivity_threshold) / len(recent_detections)
            
            if false_positive_rate > 0.1:  # Too many false positives
                self.adaptive_threshold *= 1.1
            elif false_positive_rate < 0.01:  # Too few detections, might be missing attacks
                self.adaptive_threshold *= 0.95
            
            # Keep threshold within reasonable bounds
            self.adaptive_threshold = max(0.01, min(0.9, self.adaptive_threshold))


class InputSanitizer:
    """Advanced input sanitization for neural operator inputs."""
    
    def __init__(self, max_input_norm: float = 10.0, enable_noise_filtering: bool = True):
        self.max_input_norm = max_input_norm
        self.enable_noise_filtering = enable_noise_filtering
        
        # Sanitization statistics
        self.sanitization_stats = {
            'inputs_processed': 0,
            'inputs_modified': 0,
            'norm_clipping_applied': 0,
            'noise_filtering_applied': 0
        }
        
        logging.info(f"Initialized Input Sanitizer with max_norm={max_input_norm}")
    
    def sanitize_input(self, input_tensor: torch.Tensor, 
                      preserve_semantics: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Sanitize input tensor to remove potential adversarial perturbations.
        
        Args:
            input_tensor: Input tensor to sanitize
            preserve_semantics: Whether to preserve semantic content
            
        Returns:
            (sanitized_tensor, sanitization_info)
        """
        sanitized = input_tensor.clone()
        sanitization_info = {
            'original_norm': torch.norm(input_tensor).item(),
            'modifications_applied': []
        }
        
        self.sanitization_stats['inputs_processed'] += 1
        
        # 1. Norm clipping
        input_norm = torch.norm(sanitized)
        if input_norm > self.max_input_norm:
            sanitized = sanitized * (self.max_input_norm / input_norm)
            sanitization_info['modifications_applied'].append('norm_clipping')
            self.sanitization_stats['norm_clipping_applied'] += 1
        
        # 2. Noise filtering (if enabled)
        if self.enable_noise_filtering:
            filtered = self._apply_noise_filter(sanitized, preserve_semantics)
            if not torch.equal(filtered, sanitized):
                sanitized = filtered
                sanitization_info['modifications_applied'].append('noise_filtering')
                self.sanitization_stats['noise_filtering_applied'] += 1
        
        # 3. Value clamping for numerical stability
        sanitized = torch.clamp(sanitized, min=-100.0, max=100.0)
        
        # 4. NaN/Inf protection
        if torch.isnan(sanitized).any() or torch.isinf(sanitized).any():
            sanitized = torch.where(torch.isnan(sanitized) | torch.isinf(sanitized),
                                  torch.zeros_like(sanitized), sanitized)
            sanitization_info['modifications_applied'].append('nan_inf_protection')
        
        sanitization_info['final_norm'] = torch.norm(sanitized).item()
        sanitization_info['modification_magnitude'] = torch.norm(input_tensor - sanitized).item()
        
        if sanitization_info['modifications_applied']:
            self.sanitization_stats['inputs_modified'] += 1
        
        return sanitized, sanitization_info
    
    def _apply_noise_filter(self, input_tensor: torch.Tensor, preserve_semantics: bool) -> torch.Tensor:
        """Apply noise filtering while preserving semantic content."""
        try:
            if not preserve_semantics:
                # Simple Gaussian filtering
                if len(input_tensor.shape) >= 2:
                    # Apply mild Gaussian blur to reduce high-frequency noise
                    kernel_size = 3
                    sigma = 0.5
                    
                    # Create Gaussian kernel
                    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
                    gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
                    gaussian_1d = gaussian_1d / gaussian_1d.sum()
                    
                    # Apply separable filtering
                    filtered = input_tensor
                    for dim in range(-2, 0):  # Apply to last two dimensions
                        filtered = torch.conv1d(
                            filtered.flatten(end_dim=-3).unsqueeze(1) if len(filtered.shape) > 2 
                            else filtered.unsqueeze(0).unsqueeze(0),
                            gaussian_1d.view(1, 1, -1),
                            padding=kernel_size//2
                        ).squeeze()
                    
                    return filtered.view(input_tensor.shape)
            
            # For semantic preservation, use more conservative filtering
            # Median filter for impulse noise
            if len(input_tensor.shape) >= 2:
                # Simple median filtering (approximate)
                kernel_size = 3
                pad_size = kernel_size // 2
                
                padded = F.pad(input_tensor, [pad_size, pad_size, pad_size, pad_size], 
                              mode='replicate')
                
                filtered = torch.zeros_like(input_tensor)
                for i in range(input_tensor.shape[-2]):
                    for j in range(input_tensor.shape[-1]):
                        patch = padded[..., i:i+kernel_size, j:j+kernel_size]
                        filtered[..., i, j] = torch.median(patch.flatten(start_dim=-2), dim=-1)[0]
                
                return filtered
            
            return input_tensor
            
        except Exception as e:
            logging.warning(f"Noise filtering failed: {e}")
            return input_tensor


class ModelIntegrityVerifier:
    """Verify model integrity and detect tampering."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.reference_hash = self._compute_model_hash()
        self.parameter_checksums = self._compute_parameter_checksums()
        
        # Integrity check history
        self.integrity_history = deque(maxlen=1000)
        
        logging.info("Initialized Model Integrity Verifier")
    
    def verify_integrity(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify current model integrity against reference."""
        current_hash = self._compute_model_hash()
        current_checksums = self._compute_parameter_checksums()
        
        # Overall integrity check
        overall_integrity = current_hash == self.reference_hash
        
        # Parameter-level integrity
        parameter_integrity = {}
        tampered_parameters = []
        
        for param_name, ref_checksum in self.parameter_checksums.items():
            current_checksum = current_checksums.get(param_name, "missing")
            is_intact = current_checksum == ref_checksum
            parameter_integrity[param_name] = is_intact
            
            if not is_intact:
                tampered_parameters.append(param_name)
        
        integrity_info = {
            'overall_integrity': overall_integrity,
            'parameter_integrity': parameter_integrity,
            'tampered_parameters': tampered_parameters,
            'reference_hash': self.reference_hash,
            'current_hash': current_hash,
            'timestamp': time.time()
        }
        
        self.integrity_history.append(integrity_info)
        
        return overall_integrity, integrity_info
    
    def update_reference(self):
        """Update reference hashes (use carefully!)."""
        logging.warning("Updating model integrity reference - ensure model is trusted")
        self.reference_hash = self._compute_model_hash()
        self.parameter_checksums = self._compute_parameter_checksums()
    
    def _compute_model_hash(self) -> str:
        """Compute SHA-256 hash of entire model state."""
        hasher = hashlib.sha256()
        
        for name, param in self.model.named_parameters():
            hasher.update(name.encode())
            hasher.update(param.data.cpu().numpy().tobytes())
        
        return hasher.hexdigest()
    
    def _compute_parameter_checksums(self) -> Dict[str, str]:
        """Compute individual parameter checksums."""
        checksums = {}
        
        for name, param in self.model.named_parameters():
            param_hasher = hashlib.sha256()
            param_hasher.update(param.data.cpu().numpy().tobytes())
            checksums[name] = param_hasher.hexdigest()
        
        return checksums


class AdaptiveSecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        # Security components
        self.adversarial_detector = AdversarialDetector()
        self.input_sanitizer = InputSanitizer()
        self.integrity_verifier = ModelIntegrityVerifier(model)
        
        # Security configuration
        self.security_level = ThreatLevel.MEDIUM
        self.auto_mitigation = True
        self.logging_enabled = True
        
        # Security incident tracking
        self.security_incidents = deque(maxlen=10000)
        self.threat_level_history = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.Lock()
        
        logging.info("Initialized Adaptive Security Manager")
    
    def secure_inference(self, input_tensor: torch.Tensor, 
                        enable_sanitization: bool = True,
                        enable_integrity_check: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform secure inference with comprehensive security checks.
        
        Returns:
            (model_output, security_report)
        """
        security_report = {
            'timestamp': time.time(),
            'input_hash': hashlib.sha256(input_tensor.cpu().numpy().tobytes()).hexdigest(),
            'security_checks': {},
            'threat_level': ThreatLevel.MINIMAL,
            'incidents': []
        }
        
        with self._lock:
            try:
                # 1. Model integrity verification
                if enable_integrity_check:
                    integrity_ok, integrity_info = self.integrity_verifier.verify_integrity()
                    security_report['security_checks']['integrity'] = integrity_info
                    
                    if not integrity_ok:
                        incident = SecurityIncident(
                            timestamp=time.time(),
                            attack_type=AttackType.MODEL_POISONING,
                            threat_level=ThreatLevel.CRITICAL,
                            confidence_score=1.0,
                            input_hash=security_report['input_hash'],
                            attack_vector={'tampered_parameters': integrity_info['tampered_parameters']},
                            mitigation_applied="integrity_verification_failed",
                            success=False
                        )
                        
                        self.security_incidents.append(incident)
                        security_report['incidents'].append(incident)
                        security_report['threat_level'] = ThreatLevel.CRITICAL
                        
                        raise SecurityError("Model integrity compromised")
                
                # 2. Adversarial input detection
                is_adversarial, confidence, detection_details = self.adversarial_detector.detect_adversarial_input(
                    input_tensor, self.model
                )
                
                security_report['security_checks']['adversarial_detection'] = {
                    'is_adversarial': is_adversarial,
                    'confidence': confidence,
                    'details': detection_details
                }
                
                processed_input = input_tensor
                
                # 3. Input sanitization (if adversarial or if enabled)
                if (is_adversarial and self.auto_mitigation) or enable_sanitization:
                    processed_input, sanitization_info = self.input_sanitizer.sanitize_input(
                        input_tensor, preserve_semantics=not is_adversarial
                    )
                    security_report['security_checks']['sanitization'] = sanitization_info
                    
                    if is_adversarial:
                        incident = SecurityIncident(
                            timestamp=time.time(),
                            attack_type=AttackType.ADVERSARIAL_INPUT,
                            threat_level=ThreatLevel.HIGH if confidence > 0.8 else ThreatLevel.MEDIUM,
                            confidence_score=confidence,
                            input_hash=security_report['input_hash'],
                            attack_vector=detection_details,
                            mitigation_applied="input_sanitization",
                            success=True
                        )
                        
                        self.security_incidents.append(incident)
                        security_report['incidents'].append(incident)
                        security_report['threat_level'] = max(security_report['threat_level'], 
                                                             incident.threat_level, key=lambda x: x.value)
                
                # 4. Secure inference
                with torch.no_grad():
                    model_output = self.model(processed_input)
                
                # 5. Output validation
                output_validation = self._validate_output(model_output)
                security_report['security_checks']['output_validation'] = output_validation
                
                if not output_validation['is_valid']:
                    security_report['threat_level'] = ThreatLevel.HIGH
                
                # 6. Update threat level history
                self.threat_level_history.append({
                    'timestamp': time.time(),
                    'threat_level': security_report['threat_level'],
                    'incident_count': len(security_report['incidents'])
                })
                
                return model_output, security_report
                
            except Exception as e:
                logging.error(f"Security error during inference: {e}")
                security_report['error'] = str(e)
                security_report['threat_level'] = ThreatLevel.CRITICAL
                raise
    
    def _validate_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """Validate model output for anomalies."""
        validation_info = {
            'is_valid': True,
            'issues': []
        }
        
        # Check for NaN/Inf values
        if torch.isnan(output).any():
            validation_info['is_valid'] = False
            validation_info['issues'].append('contains_nan')
        
        if torch.isinf(output).any():
            validation_info['is_valid'] = False
            validation_info['issues'].append('contains_inf')
        
        # Check for extreme values
        output_norm = torch.norm(output).item()
        if output_norm > 1000.0:
            validation_info['is_valid'] = False
            validation_info['issues'].append('extreme_magnitude')
        
        # Check for zero/constant outputs (potential attack sign)
        if torch.allclose(output, torch.zeros_like(output), atol=1e-10):
            validation_info['issues'].append('zero_output')
        elif torch.allclose(output, output[0], atol=1e-10):
            validation_info['issues'].append('constant_output')
        
        validation_info['output_norm'] = output_norm
        validation_info['output_range'] = [float(torch.min(output)), float(torch.max(output))]
        
        return validation_info
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Generate comprehensive security summary."""
        current_time = time.time()
        
        # Recent incidents (last hour)
        recent_incidents = [
            incident for incident in self.security_incidents
            if current_time - incident.timestamp < 3600
        ]
        
        # Threat level distribution
        threat_distribution = {}
        for level in ThreatLevel:
            threat_distribution[level.value] = sum(
                1 for incident in recent_incidents
                if incident.threat_level == level
            )
        
        # Attack type distribution
        attack_distribution = {}
        for attack_type in AttackType:
            attack_distribution[attack_type.value] = sum(
                1 for incident in recent_incidents
                if incident.attack_type == attack_type
            )
        
        # Security health score
        if recent_incidents:
            avg_threat_level = sum(
                list(ThreatLevel).index(incident.threat_level) for incident in recent_incidents
            ) / len(recent_incidents)
            security_health = max(0.0, 1.0 - avg_threat_level / len(ThreatLevel))
        else:
            security_health = 1.0
        
        return {
            'security_health_score': security_health,
            'current_threat_level': self.security_level.value,
            'recent_incidents_count': len(recent_incidents),
            'total_incidents_count': len(self.security_incidents),
            'threat_level_distribution': threat_distribution,
            'attack_type_distribution': attack_distribution,
            'detector_stats': {
                'adaptive_threshold': self.adversarial_detector.adaptive_threshold,
                'detection_history_size': len(self.adversarial_detector.detection_history)
            },
            'sanitizer_stats': self.input_sanitizer.sanitization_stats,
            'integrity_status': 'verified' if len(self.integrity_verifier.integrity_history) == 0 
                               or self.integrity_verifier.integrity_history[-1]['overall_integrity']
                               else 'compromised'
        }


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


# Export classes
__all__ = [
    'ThreatLevel',
    'AttackType', 
    'SecurityIncident',
    'AdversarialDetector',
    'InputSanitizer',
    'ModelIntegrityVerifier',
    'AdaptiveSecurityManager',
    'SecurityError'
]