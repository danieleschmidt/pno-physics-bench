#!/bin/bash
set -e

CONFIG_FILE=${1:-production.json}
NAMESPACE=${2:-default}

echo "Deploying PNO Physics Bench with config: $CONFIG_FILE"

# Apply Kubernetes configuration
kubectl apply -f configs/kubernetes.yaml -n $NAMESPACE

# Wait for deployment to be ready
kubectl wait --for=condition=available --timeout=300s deployment/pno-physics-bench -n $NAMESPACE

# Verify deployment
kubectl get pods -l app=pno-physics-bench -n $NAMESPACE
kubectl get services -l app=pno-physics-bench -n $NAMESPACE

echo "Deployment completed successfully!"
