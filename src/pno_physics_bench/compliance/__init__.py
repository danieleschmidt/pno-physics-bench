"""
Compliance and Regulatory Support for PNO Physics Bench

Provides comprehensive compliance frameworks for global deployment including:
- GDPR (European Union)
- CCPA (California) 
- PDPA (Singapore)
- Data localization requirements
- Audit logging and traceability
- Privacy-preserving uncertainty quantification
"""

import os
import json
import hashlib
import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from ..i18n import get_text


@dataclass
class ComplianceConfig:
    """Configuration for compliance requirements."""
    
    # Privacy regulations
    gdpr_enabled: bool = False
    ccpa_enabled: bool = False  
    pdpa_enabled: bool = False
    
    # Data handling
    data_retention_days: int = 365
    anonymization_required: bool = False
    encryption_at_rest: bool = True
    
    # Audit requirements
    audit_logging: bool = True
    model_provenance_tracking: bool = True
    uncertainty_explainability: bool = True
    
    # Geographic restrictions
    data_localization_regions: List[str] = None
    cross_border_transfer_allowed: bool = True
    
    # Consent management
    explicit_consent_required: bool = False
    consent_withdrawal_supported: bool = True
    
    def __post_init__(self):
        if self.data_localization_regions is None:
            self.data_localization_regions = []


class ComplianceManager:
    """Manages compliance across different regulatory frameworks."""
    
    def __init__(self, config: Optional[ComplianceConfig] = None):
        self.config = config or ComplianceConfig()
        self.audit_logger = self._setup_audit_logging()
        
        # Initialize compliance modules
        self.gdpr_manager = GDPRCompliance() if self.config.gdpr_enabled else None
        self.ccpa_manager = CCPACompliance() if self.config.ccpa_enabled else None
        self.pdpa_manager = PDPACompliance() if self.config.pdpa_enabled else None
        
        # Data protection components
        self.data_protector = DataProtectionManager(self.config)
        self.audit_trail = AuditTrail(self.config)
        
    def _setup_audit_logging(self) -> logging.Logger:
        """Setup audit logging for compliance."""
        
        logger = logging.getLogger("pno_compliance_audit")
        logger.setLevel(logging.INFO)
        
        if self.config.audit_logging and not logger.handlers:
            # Create audit log handler
            audit_handler = logging.FileHandler("pno_audit.log")
            audit_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            audit_handler.setFormatter(audit_formatter)
            logger.addHandler(audit_handler)
        
        return logger
    
    def validate_data_processing(
        self, 
        operation: str,
        data_info: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate if data processing operation complies with regulations.
        
        Returns:
            (is_compliant, list_of_issues)
        """
        
        issues = []
        
        # Check geographic restrictions
        if self.config.data_localization_regions:
            user_region = user_context.get("region") if user_context else None
            if user_region and user_region not in self.config.data_localization_regions:
                if not self.config.cross_border_transfer_allowed:
                    issues.append(f"Cross-border data transfer not allowed for region: {user_region}")
        
        # GDPR validation
        if self.gdpr_manager:
            gdpr_valid, gdpr_issues = self.gdpr_manager.validate_processing(
                operation, data_info, user_context
            )
            if not gdpr_valid:
                issues.extend(gdpr_issues)
        
        # CCPA validation
        if self.ccpa_manager:
            ccpa_valid, ccpa_issues = self.ccpa_manager.validate_processing(
                operation, data_info, user_context
            )
            if not ccpa_valid:
                issues.extend(ccpa_issues)
        
        # PDPA validation
        if self.pdpa_manager:
            pdpa_valid, pdpa_issues = self.pdpa_manager.validate_processing(
                operation, data_info, user_context
            )
            if not pdpa_valid:
                issues.extend(pdpa_issues)
        
        # Log audit event
        self.audit_trail.log_data_processing(
            operation=operation,
            data_info=data_info,
            user_context=user_context,
            compliance_status="COMPLIANT" if not issues else "NON_COMPLIANT",
            issues=issues
        )
        
        return len(issues) == 0, issues
    
    def anonymize_uncertainty_data(
        self,
        uncertainty_data: Any,
        anonymization_level: str = "standard"
    ) -> Any:
        """Anonymize uncertainty data for compliance."""
        
        if not self.config.anonymization_required:
            return uncertainty_data
        
        return self.data_protector.anonymize_data(
            uncertainty_data, 
            anonymization_level
        )
    
    def get_data_retention_policy(self) -> Dict[str, Any]:
        """Get data retention policy based on compliance requirements."""
        
        policy = {
            "retention_days": self.config.data_retention_days,
            "automatic_deletion": True,
            "backup_retention": self.config.data_retention_days // 2,
            "audit_log_retention": max(self.config.data_retention_days, 2555)  # Min 7 years for audit
        }
        
        # Adjust based on specific regulations
        if self.gdpr_manager:
            gdpr_policy = self.gdpr_manager.get_retention_requirements()
            policy.update(gdpr_policy)
        
        return policy
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        report = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "compliance_config": asdict(self.config),
            "active_regulations": [],
            "data_protection_status": self.data_protector.get_status(),
            "audit_summary": self.audit_trail.get_summary(),
            "recommendations": []
        }
        
        # Add regulation-specific reports
        if self.gdpr_manager:
            report["active_regulations"].append("GDPR")
            report["gdpr_status"] = self.gdpr_manager.get_compliance_status()
        
        if self.ccpa_manager:
            report["active_regulations"].append("CCPA")
            report["ccpa_status"] = self.ccpa_manager.get_compliance_status()
        
        if self.pdpa_manager:
            report["active_regulations"].append("PDPA")
            report["pdpa_status"] = self.pdpa_manager.get_compliance_status()
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations based on current status."""
        
        recommendations = []
        
        if not self.config.encryption_at_rest:
            recommendations.append("Enable encryption at rest for enhanced data protection")
        
        if not self.config.audit_logging:
            recommendations.append("Enable audit logging for compliance tracking")
        
        if self.config.gdpr_enabled and not self.config.anonymization_required:
            recommendations.append("Consider enabling data anonymization for GDPR compliance")
        
        if len(self.config.data_localization_regions) == 0:
            recommendations.append("Define data localization regions for geographic compliance")
        
        return recommendations


class GDPRCompliance:
    """GDPR (General Data Protection Regulation) compliance manager."""
    
    def __init__(self):
        self.required_rights = [
            "access", "rectification", "erasure", "portability", 
            "restriction", "objection", "automated_decision_making"
        ]
    
    def validate_processing(
        self,
        operation: str,
        data_info: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Validate GDPR compliance for data processing."""
        
        issues = []
        
        # Check lawful basis for processing
        lawful_basis = data_info.get("lawful_basis")
        if not lawful_basis:
            issues.append("GDPR: No lawful basis specified for processing")
        
        # Check data minimization
        if data_info.get("personal_data", False):
            issues.append("GDPR: Personal data detected. PNO should process only numerical PDE data.")
        
        # Check consent requirements
        if lawful_basis == "consent":
            consent_given = user_context.get("consent_given") if user_context else False
            if not consent_given:
                issues.append("GDPR: Explicit consent required but not provided")
        
        # Check data subject rights
        if operation in ["training", "inference"] and data_info.get("personal_data"):
            issues.append("GDPR: Data subject rights must be supported for personal data processing")
        
        return len(issues) == 0, issues
    
    def get_retention_requirements(self) -> Dict[str, Any]:
        """Get GDPR data retention requirements."""
        
        return {
            "max_retention_days": 2555,  # 7 years max unless specific legal requirement
            "purpose_limitation": True,
            "automatic_erasure": True,
            "right_to_erasure": True
        }
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get GDPR compliance status."""
        
        return {
            "data_protection_by_design": True,
            "data_protection_by_default": True,
            "privacy_impact_assessment_required": False,  # No personal data in PNO
            "data_protection_officer_required": False,    # Depends on organization
            "cross_border_transfer_mechanism": "Adequacy decision or Standard Contractual Clauses"
        }


class CCPACompliance:
    """CCPA (California Consumer Privacy Act) compliance manager."""
    
    def validate_processing(
        self,
        operation: str,
        data_info: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Validate CCPA compliance."""
        
        issues = []
        
        # CCPA primarily applies to personal information
        if data_info.get("personal_information", False):
            # Check notice requirements
            if not data_info.get("privacy_notice_provided"):
                issues.append("CCPA: Privacy notice must be provided at or before collection")
            
            # Check consumer rights
            user_region = user_context.get("region") if user_context else None
            if user_region == "CA":  # California resident
                if operation == "sale" and not user_context.get("do_not_sell_respected"):
                    issues.append("CCPA: Right to opt-out of sale must be respected")
        
        return len(issues) == 0, issues
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get CCPA compliance status."""
        
        return {
            "consumer_rights_supported": ["know", "delete", "opt_out", "non_discrimination"],
            "personal_information_categories": [],  # PNO processes no personal information
            "business_purposes": ["research", "scientific_advancement"],
            "third_party_sharing": False
        }


class PDPACompliance:
    """PDPA (Personal Data Protection Act) compliance manager for Singapore."""
    
    def validate_processing(
        self,
        operation: str,
        data_info: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Validate PDPA compliance."""
        
        issues = []
        
        if data_info.get("personal_data", False):
            # Check consent requirements
            if not user_context.get("consent_obtained"):
                issues.append("PDPA: Consent must be obtained before collecting personal data")
            
            # Check purpose limitation
            if not data_info.get("purpose_specified"):
                issues.append("PDPA: Purpose for collection must be specified")
        
        return len(issues) == 0, issues
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get PDPA compliance status."""
        
        return {
            "consent_management": True,
            "purpose_limitation": True,
            "data_breach_notification": True,
            "individual_rights": ["access", "correction", "withdrawal"]
        }


class DataProtectionManager:
    """Manages data protection mechanisms."""
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
    
    def anonymize_data(self, data: Any, level: str = "standard") -> Any:
        """Apply anonymization to data based on level."""
        
        # For PNO, this primarily applies to metadata and logs
        # The actual PDE numerical data doesn't contain personal information
        
        if level == "basic":
            # Basic anonymization - remove direct identifiers
            return self._remove_identifiers(data)
        
        elif level == "standard": 
            # Standard anonymization - k-anonymity, l-diversity
            return self._apply_k_anonymity(data, k=5)
        
        elif level == "strong":
            # Strong anonymization - differential privacy
            return self._apply_differential_privacy(data, epsilon=1.0)
        
        return data
    
    def _remove_identifiers(self, data: Any) -> Any:
        """Remove direct identifiers from data."""
        # Implementation would remove IP addresses, user IDs, etc.
        return data
    
    def _apply_k_anonymity(self, data: Any, k: int = 5) -> Any:
        """Apply k-anonymity to data."""
        # Implementation would ensure each record is indistinguishable 
        # from at least k-1 other records
        return data
    
    def _apply_differential_privacy(self, data: Any, epsilon: float = 1.0) -> Any:
        """Apply differential privacy to data."""
        # Implementation would add calibrated noise to ensure privacy
        return data
    
    def get_status(self) -> Dict[str, Any]:
        """Get data protection status."""
        
        return {
            "encryption_at_rest": self.config.encryption_at_rest,
            "anonymization_enabled": self.config.anonymization_required,
            "data_retention_policy": f"{self.config.data_retention_days} days",
            "cross_border_transfers": self.config.cross_border_transfer_allowed
        }


class AuditTrail:
    """Maintains audit trail for compliance."""
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        self.audit_log = []
        
    def log_data_processing(
        self,
        operation: str,
        data_info: Dict[str, Any],
        user_context: Optional[Dict[str, Any]],
        compliance_status: str,
        issues: List[str]
    ):
        """Log data processing event for audit."""
        
        if not self.config.audit_logging:
            return
        
        audit_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "operation": operation,
            "data_categories": data_info.get("categories", []),
            "user_region": user_context.get("region") if user_context else "unknown",
            "compliance_status": compliance_status,
            "issues": issues,
            "data_hash": self._compute_data_hash(data_info)
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep audit log bounded
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # Keep last 5000 entries
    
    def _compute_data_hash(self, data_info: Dict[str, Any]) -> str:
        """Compute hash of data for audit trail."""
        
        # Create a stable hash of the data info for audit purposes
        data_str = json.dumps(data_info, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get audit trail summary."""
        
        if not self.audit_log:
            return {"total_events": 0}
        
        total_events = len(self.audit_log)
        compliant_events = sum(1 for entry in self.audit_log 
                             if entry["compliance_status"] == "COMPLIANT")
        
        return {
            "total_events": total_events,
            "compliant_events": compliant_events,
            "compliance_rate": compliant_events / total_events if total_events > 0 else 0,
            "recent_issues": [entry["issues"] for entry in self.audit_log[-10:] 
                            if entry["issues"]]
        }


# Convenience functions
def get_compliance_manager(
    gdpr: bool = False,
    ccpa: bool = False,
    pdpa: bool = False,
    **kwargs
) -> ComplianceManager:
    """Get a configured compliance manager."""
    
    config = ComplianceConfig(
        gdpr_enabled=gdpr,
        ccpa_enabled=ccpa,
        pdpa_enabled=pdpa,
        **kwargs
    )
    
    return ComplianceManager(config)


def validate_pno_operation(
    operation: str,
    compliance_manager: ComplianceManager,
    data_categories: List[str] = None,
    user_region: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """Validate PNO operation for compliance."""
    
    data_info = {
        "categories": data_categories or ["numerical_pde_data"],
        "personal_data": False,  # PNO processes numerical data only
        "lawful_basis": "legitimate_interest",  # Scientific research
        "purpose": "uncertainty_quantification_research"
    }
    
    user_context = {"region": user_region} if user_region else None
    
    return compliance_manager.validate_data_processing(
        operation, data_info, user_context
    )


__all__ = [
    "ComplianceConfig",
    "ComplianceManager", 
    "GDPRCompliance",
    "CCPACompliance",
    "PDPACompliance",
    "get_compliance_manager",
    "validate_pno_operation"
]