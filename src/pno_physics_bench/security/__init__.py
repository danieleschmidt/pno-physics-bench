# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Security modules for PNO Physics Bench."""

from .audit_logging import (
    AuditLevel,
    EventType,
    AuditEvent,
    AuditLogger,
    SecurityMonitor,
    audit_decorator,
    global_audit_logger,
    set_global_audit_logger,
    get_global_audit_logger
)

__all__ = [
    'AuditLevel',
    'EventType', 
    'AuditEvent',
    'AuditLogger',
    'SecurityMonitor',
    'audit_decorator',
    'global_audit_logger',
    'set_global_audit_logger',
    'get_global_audit_logger'
]