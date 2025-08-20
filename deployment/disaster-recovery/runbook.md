# Disaster Recovery Runbook

## Overview
This runbook provides step-by-step procedures for disaster recovery scenarios.

## Scenarios

### 1. Complete Data Center Outage
1. **Assessment**: Verify outage scope and expected duration
2. **Communication**: Notify stakeholders via status page
3. **Failover**: Switch traffic to backup region
   ```bash
   kubectl apply -f deployment/disaster-recovery/failover.yaml
   ```
4. **Verification**: Confirm service availability in backup region
5. **Monitoring**: Monitor backup region performance

### 2. Database Corruption
1. **Stop Application**: Scale deployment to 0 replicas
   ```bash
   kubectl scale deployment pno-physics-bench --replicas=0
   ```
2. **Restore Database**: Restore from latest backup
   ```bash
   ./scripts/restore-database.sh <backup-timestamp>
   ```
3. **Validate Data**: Run data integrity checks
4. **Restart Application**: Scale deployment back up
   ```bash
   kubectl scale deployment pno-physics-bench --replicas=3
   ```

### 3. Security Incident
1. **Isolate**: Immediately isolate affected systems
2. **Assess**: Determine scope of compromise
3. **Contain**: Stop the attack and prevent spread
4. **Eradicate**: Remove malicious artifacts
5. **Recover**: Restore systems from clean backups
6. **Learn**: Conduct post-incident review

## Recovery Time Objectives (RTO)
- Critical services: 15 minutes
- Non-critical services: 1 hour
- Full system recovery: 4 hours

## Recovery Point Objectives (RPO)
- Database: 1 hour (hourly backups)
- Configuration: 15 minutes (real-time sync)
- Code: 0 (version controlled)

## Emergency Contacts
- On-call Engineer: +1-XXX-XXX-XXXX
- Security Team: security@terragonlabs.com
- Infrastructure Team: infra@terragonlabs.com
