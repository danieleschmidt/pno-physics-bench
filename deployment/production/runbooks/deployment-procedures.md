# PNO Physics Bench - Production Deployment Procedures

## Overview
This runbook provides comprehensive procedures for deploying, maintaining, and troubleshooting the PNO Physics Bench production environment.

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Procedures](#deployment-procedures)
3. [Zero-Downtime Deployment](#zero-downtime-deployment)
4. [Rollback Procedures](#rollback-procedures)
5. [Health Checks and Validation](#health-checks-and-validation)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Maintenance Procedures](#maintenance-procedures)

## Pre-Deployment Checklist

### Infrastructure Requirements
- [ ] Kubernetes cluster version 1.20+ running
- [ ] Helm 3.7+ installed and configured
- [ ] kubectl configured with cluster admin access
- [ ] Container registry access (ghcr.io)
- [ ] DNS records configured for ingress endpoints
- [ ] TLS certificates provisioned
- [ ] Persistent storage provisioned (100Gi per region)

### Environment Preparation
- [ ] Production namespace created: `kubectl create namespace production`
- [ ] Monitoring namespace created: `kubectl create namespace monitoring`
- [ ] Service accounts and RBAC configured
- [ ] Network policies applied
- [ ] Pod security policies configured
- [ ] Resource quotas set

### Security Validation
- [ ] Image security scan passed
- [ ] Vulnerability assessment completed
- [ ] Compliance requirements verified
- [ ] Access controls validated
- [ ] Secrets management configured

### Performance Validation
- [ ] Load testing completed successfully
- [ ] Performance benchmarks met
- [ ] Resource requirements validated
- [ ] Auto-scaling thresholds tested

## Deployment Procedures

### Standard Deployment Process

#### 1. Prepare Deployment Environment
```bash
#!/bin/bash
# Set deployment variables
export DEPLOYMENT_VERSION=v1.0.0
export ENVIRONMENT=production
export NAMESPACE=production
export IMAGE_TAG=${DEPLOYMENT_VERSION}
export REGISTRY=ghcr.io

# Validate cluster connection
kubectl cluster-info
kubectl get nodes
```

#### 2. Deploy Infrastructure Components
```bash
# Deploy security policies first
kubectl apply -f deployment/production/security/

# Deploy monitoring infrastructure
kubectl apply -f deployment/production/monitoring/

# Deploy storage and configuration
kubectl apply -f deployment/production/storage/
kubectl apply -f deployment/production/config/
```

#### 3. Deploy Application
```bash
# Deploy main application
kubectl apply -f deployment/production/deployment.yaml
kubectl apply -f deployment/production/service.yaml
kubectl apply -f deployment/production/hpa.yaml
kubectl apply -f deployment/production/ingress.yaml

# Verify deployment
kubectl rollout status deployment/pno-physics-bench -n production --timeout=300s
```

#### 4. Post-Deployment Validation
```bash
# Check pod status
kubectl get pods -n production -l app=pno-physics-bench

# Verify service endpoints
kubectl get svc -n production pno-service

# Test health endpoints
curl -f https://api.pno-physics-bench.com/health
curl -f https://api.pno-physics-bench.com/ready
```

## Zero-Downtime Deployment

### Blue-Green Deployment Strategy

#### 1. Prepare Blue Environment (Current Production)
```bash
# Label current deployment as blue
kubectl label deployment pno-physics-bench environment=blue -n production
```

#### 2. Deploy Green Environment (New Version)
```bash
# Create green deployment manifest
cat > deployment-green.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pno-physics-bench-green
  namespace: production
  labels:
    app: pno-physics-bench
    environment: green
    version: ${NEW_VERSION}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pno-physics-bench
      environment: green
  template:
    metadata:
      labels:
        app: pno-physics-bench
        environment: green
        version: ${NEW_VERSION}
    spec:
      # ... (same spec as blue with new image version)
EOF

# Deploy green environment
kubectl apply -f deployment-green.yaml
kubectl rollout status deployment/pno-physics-bench-green -n production
```

#### 3. Switch Traffic to Green
```bash
# Update service selector to point to green
kubectl patch service pno-service -n production -p '{"spec":{"selector":{"environment":"green"}}}'

# Verify traffic switch
kubectl describe service pno-service -n production
```

#### 4. Validate Green Deployment
```bash
# Run health checks
curl -f https://api.pno-physics-bench.com/health
curl -f https://api.pno-physics-bench.com/ready

# Run integration tests
python tests/integration/test_production_api.py

# Monitor for 5 minutes for any issues
sleep 300
```

#### 5. Cleanup Blue Environment
```bash
# If green is stable, remove blue deployment
kubectl delete deployment pno-physics-bench -n production
```

### Rolling Update Deployment
```bash
# For standard rolling updates
kubectl set image deployment/pno-physics-bench pno-inference=pno-physics-bench:${NEW_VERSION} -n production

# Monitor rollout
kubectl rollout status deployment/pno-physics-bench -n production

# Verify new version
kubectl get pods -n production -l app=pno-physics-bench -o jsonpath='{.items[*].spec.containers[0].image}'
```

## Rollback Procedures

### Immediate Rollback (Emergency)
```bash
# Get rollout history
kubectl rollout history deployment/pno-physics-bench -n production

# Rollback to previous version
kubectl rollout undo deployment/pno-physics-bench -n production

# Monitor rollback
kubectl rollout status deployment/pno-physics-bench -n production
```

### Specific Version Rollback
```bash
# Rollback to specific revision
kubectl rollout undo deployment/pno-physics-bench --to-revision=2 -n production

# Verify rollback
kubectl get pods -n production -l app=pno-physics-bench
```

### Blue-Green Rollback
```bash
# Switch service back to blue environment
kubectl patch service pno-service -n production -p '{"spec":{"selector":{"environment":"blue"}}}'

# Verify rollback
kubectl describe service pno-service -n production
```

## Health Checks and Validation

### Application Health Checks
```bash
# Health endpoint check
curl -f https://api.pno-physics-bench.com/health

# Readiness endpoint check
curl -f https://api.pno-physics-bench.com/ready

# Metrics endpoint check
curl -f https://api.pno-physics-bench.com/metrics
```

### Infrastructure Health Checks
```bash
# Check pod status
kubectl get pods -n production -l app=pno-physics-bench

# Check resource usage
kubectl top pods -n production -l app=pno-physics-bench

# Check events
kubectl get events -n production --sort-by='.lastTimestamp' | head -20
```

### Performance Validation
```bash
# Run load test
python scripts/load_test.py --endpoint https://api.pno-physics-bench.com --duration 300

# Check response times
curl -w "@curl-format.txt" -s -o /dev/null https://api.pno-physics-bench.com/predict
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Pod Startup Issues
**Symptoms**: Pods stuck in Pending/ContainerCreating state
**Investigation**:
```bash
kubectl describe pod <pod-name> -n production
kubectl logs <pod-name> -n production
```
**Solutions**:
- Check resource availability: `kubectl describe nodes`
- Verify image pull: `kubectl get events -n production`
- Check security policies: `kubectl get psp`

#### High Memory Usage
**Symptoms**: Memory usage exceeding limits
**Investigation**:
```bash
kubectl top pods -n production
kubectl describe pod <pod-name> -n production
```
**Solutions**:
- Scale up resources: Update deployment resource limits
- Optimize model loading: Check memory-efficient model loading
- Add memory profiling: Enable memory monitoring

#### Network Connectivity Issues
**Symptoms**: Service unavailable, connection timeouts
**Investigation**:
```bash
kubectl get svc -n production
kubectl describe ingress pno-ingress -n production
kubectl get networkpolicy -n production
```
**Solutions**:
- Check service endpoints: `kubectl get endpoints -n production`
- Verify network policies: Review ingress/egress rules
- Test internal connectivity: Run network debug pod

#### Performance Degradation
**Symptoms**: High latency, slow responses
**Investigation**:
```bash
# Check metrics
curl https://api.pno-physics-bench.com/metrics | grep inference_duration

# Check resource usage
kubectl top pods -n production

# Check HPA status
kubectl get hpa -n production
```
**Solutions**:
- Scale horizontally: Adjust HPA settings
- Optimize inference: Review model optimization
- Check dependencies: Verify external service performance

## Maintenance Procedures

### Scheduled Maintenance

#### 1. Pre-Maintenance Checks
```bash
# Backup current configuration
kubectl get all -n production -o yaml > backup-$(date +%Y%m%d).yaml

# Check cluster health
kubectl get nodes
kubectl get pods --all-namespaces | grep -E "(Error|CrashLoopBackOff|Pending)"
```

#### 2. Maintenance Window Process
```bash
# Scale down to minimum replicas
kubectl scale deployment pno-physics-bench --replicas=1 -n production

# Perform maintenance tasks
# ... maintenance activities ...

# Scale back up
kubectl scale deployment pno-physics-bench --replicas=3 -n production
```

#### 3. Post-Maintenance Validation
```bash
# Verify all pods are running
kubectl get pods -n production -l app=pno-physics-bench

# Run health checks
curl -f https://api.pno-physics-bench.com/health

# Run integration tests
python tests/integration/test_production_api.py
```

### Certificate Management
```bash
# Check certificate expiry
kubectl get certificates -n production

# Renew certificates (if using cert-manager)
kubectl delete certificate pno-tls-cert -n production
kubectl apply -f deployment/production/certificates/
```

### Log Management
```bash
# Archive old logs
kubectl logs --previous pno-physics-bench-<pod-id> -n production > logs/archive/

# Rotate logs
kubectl exec deployment/pno-physics-bench -n production -- logrotate /etc/logrotate.conf
```

### Security Updates
```bash
# Update base images
docker pull python:3.9-slim
docker build -t pno-physics-bench:security-update .

# Deploy security update
kubectl set image deployment/pno-physics-bench pno-inference=pno-physics-bench:security-update -n production
```

## Emergency Procedures

### Service Outage Response
1. **Immediate Response** (0-5 minutes):
   - Check service status: `kubectl get pods -n production`
   - Check ingress: `kubectl get ingress -n production`
   - Check recent changes: `kubectl get events -n production`

2. **Investigation** (5-15 minutes):
   - Review logs: `kubectl logs -l app=pno-physics-bench -n production`
   - Check metrics: Access Grafana dashboard
   - Verify dependencies: Check external services

3. **Recovery** (15-30 minutes):
   - Scale up replicas if needed
   - Rollback if deployment issue
   - Restart services if configuration issue

### Data Recovery
```bash
# Restore from backup
kubectl apply -f backups/latest-backup.yaml

# Verify data integrity
python scripts/verify_data_integrity.py
```

### Security Incident Response
1. **Isolation**: Scale down affected pods
2. **Investigation**: Collect logs and forensic data
3. **Remediation**: Apply security patches
4. **Recovery**: Redeploy clean environment

## Monitoring and Alerting

### Key Metrics to Monitor
- Response time (95th percentile < 200ms)
- Error rate (< 0.1%)
- CPU utilization (< 70% average)
- Memory utilization (< 80% average)
- Pod availability (>= 2 replicas always)

### Alert Escalation
1. **Warning Alerts**: Email to on-call team
2. **Critical Alerts**: SMS + Email + Slack
3. **Emergency**: Phone call to incident commander

### Dashboard URLs
- Production Overview: https://grafana.company.com/d/pno-production
- Infrastructure: https://grafana.company.com/d/kubernetes-cluster
- Application Metrics: https://grafana.company.com/d/pno-application

---

## Contact Information
- **On-Call Team**: +1-555-ON-CALL (665-2255)
- **Incident Commander**: +1-555-INCIDENT
- **Slack Channel**: #pno-production-alerts
- **Email**: pno-ops@company.com

## Document Version
- **Version**: 1.0.0
- **Last Updated**: 2025-08-23
- **Next Review**: 2025-11-23