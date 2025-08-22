#!/bin/bash
set -e

echo "🚀 Deploying PNO Physics Bench to Production"

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo "❌ helm not found. Please install helm."
    exit 1
fi

# Deploy namespace
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    name: production
EOF

# Deploy security policies
echo "🔒 Deploying security policies..."
kubectl apply -f security/

# Deploy monitoring
echo "📊 Deploying monitoring..."
kubectl apply -f monitoring/

# Deploy application
echo "🏗️  Deploying application..."
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml
kubectl apply -f network-policy.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/pno-physics-bench -n production --timeout=300s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n production -l app=pno-physics-bench
kubectl get svc -n production pno-service
kubectl get ingress -n production pno-ingress

echo "🎉 Deployment completed successfully!"
echo "🌐 Access the service at: https://api.pno-physics-bench.com"
