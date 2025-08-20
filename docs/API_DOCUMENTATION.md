# PNO Physics Bench - API Documentation

## Overview

The PNO Physics Bench provides a comprehensive REST API for training, inference, and management of Probabilistic Neural Operators. This API enables uncertainty quantification in neural PDE solvers through a simple yet powerful interface.

## Base URL

```
Production: https://api.pno.terragonlabs.com/v1
Staging: https://staging-api.pno.terragonlabs.com/v1
Development: http://localhost:8000/v1
```

## Authentication

### API Key Authentication
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.pno.terragonlabs.com/v1/models
```

### JWT Token Authentication
```bash
curl -H "Authorization: JWT YOUR_JWT_TOKEN" \
     https://api.pno.terragonlabs.com/v1/models
```

## Common Response Format

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2025-08-20T23:22:56.673Z",
  "request_id": "req_abc123"
}
```

Error responses:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": { ... }
  },
  "timestamp": "2025-08-20T23:22:56.673Z",
  "request_id": "req_abc123"
}
```

## Endpoints

### Models

#### List Models
```http
GET /v1/models
```

**Response:**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "id": "pno-navier-stokes-v1",
        "name": "PNO Navier-Stokes",
        "type": "probabilistic_neural_operator",
        "version": "1.0.0",
        "status": "active",
        "created_at": "2025-08-20T10:00:00Z",
        "updated_at": "2025-08-20T15:30:00Z",
        "metrics": {
          "rmse": 0.085,
          "nll": -2.31,
          "coverage_90": 89.3
        }
      }
    ],
    "total": 1,
    "page": 1,
    "per_page": 10
  }
}
```

#### Get Model Details
```http
GET /v1/models/{model_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "pno-navier-stokes-v1",
    "name": "PNO Navier-Stokes",
    "type": "probabilistic_neural_operator",
    "version": "1.0.0",
    "status": "active",
    "configuration": {
      "input_dim": 3,
      "hidden_dim": 256,
      "num_layers": 4,
      "modes": 20,
      "uncertainty_type": "full"
    },
    "training_info": {
      "dataset": "navier_stokes_2d",
      "epochs": 100,
      "batch_size": 32,
      "learning_rate": 1e-3
    },
    "metrics": {
      "rmse": 0.085,
      "nll": -2.31,
      "coverage_90": 89.3,
      "ece": 0.045
    }
  }
}
```

### Predictions

#### Single Prediction
```http
POST /v1/models/{model_id}/predict
```

**Request:**
```json
{
  "input": {
    "initial_condition": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "parameters": {
      "reynolds_number": 100,
      "time_steps": 50
    }
  },
  "options": {
    "num_samples": 100,
    "confidence_levels": [0.9, 0.95, 0.99],
    "return_uncertainty": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction": {
      "mean": [[0.8, 1.9, 2.7], [3.6, 4.8, 5.9]],
      "std": [[0.1, 0.2, 0.15], [0.18, 0.12, 0.14]]
    },
    "uncertainty": {
      "aleatoric": 0.023,
      "epistemic": 0.045,
      "total": 0.068
    },
    "confidence_intervals": {
      "90": {
        "lower": [[0.65, 1.67, 2.52], [3.42, 4.68, 5.76]],
        "upper": [[0.95, 2.13, 2.88], [3.78, 4.92, 6.04]]
      },
      "95": {
        "lower": [[0.61, 1.58, 2.45], [3.38, 4.64, 5.72]],
        "upper": [[0.99, 2.22, 2.95], [3.82, 4.96, 6.08]]
      }
    },
    "metadata": {
      "computation_time": 0.156,
      "num_samples_used": 100,
      "cache_hit": false
    }
  }
}
```

#### Batch Prediction
```http
POST /v1/models/{model_id}/predict/batch
```

**Request:**
```json
{
  "inputs": [
    {
      "id": "input_1",
      "data": {
        "initial_condition": [[1.0, 2.0, 3.0]],
        "parameters": {"reynolds_number": 100}
      }
    },
    {
      "id": "input_2", 
      "data": {
        "initial_condition": [[2.0, 3.0, 4.0]],
        "parameters": {"reynolds_number": 200}
      }
    }
  ],
  "options": {
    "num_samples": 50,
    "confidence_levels": [0.9],
    "parallel": true
  }
}
```

### Training

#### Start Training Job
```http
POST /v1/training/jobs
```

**Request:**
```json
{
  "model_config": {
    "name": "pno-custom-model",
    "type": "probabilistic_neural_operator",
    "architecture": {
      "input_dim": 3,
      "hidden_dim": 512,
      "num_layers": 6,
      "modes": 32
    }
  },
  "training_config": {
    "dataset": "custom_pde_data",
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "validation_split": 0.2
  },
  "uncertainty_config": {
    "type": "variational",
    "kl_weight": 1e-4,
    "num_samples": 10
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "job_train_abc123",
    "status": "queued",
    "estimated_duration": "2h 30m",
    "created_at": "2025-08-20T23:22:56.673Z"
  }
}
```

#### Get Training Job Status
```http
GET /v1/training/jobs/{job_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "job_train_abc123",
    "status": "training",
    "progress": 0.65,
    "current_epoch": 130,
    "total_epochs": 200,
    "metrics": {
      "train_loss": 0.0234,
      "val_loss": 0.0267,
      "train_nll": -2.45,
      "val_nll": -2.38
    },
    "elapsed_time": "1h 45m",
    "estimated_remaining": "45m"
  }
}
```

### Datasets

#### List Datasets
```http
GET /v1/datasets
```

#### Upload Dataset
```http
POST /v1/datasets
Content-Type: multipart/form-data
```

### Benchmarks

#### List Available Benchmarks
```http
GET /v1/benchmarks
```

#### Run Benchmark
```http
POST /v1/benchmarks/{benchmark_id}/run
```

### System

#### Health Check
```http
GET /v1/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": "5d 14h 32m",
    "dependencies": {
      "database": "healthy",
      "cache": "healthy",
      "ml_backend": "healthy"
    }
  }
}
```

#### System Metrics
```http
GET /v1/metrics
```

**Response:**
```json
{
  "success": true,
  "data": {
    "requests_per_second": 450,
    "average_response_time": 145,
    "error_rate": 0.002,
    "active_models": 12,
    "queued_jobs": 3,
    "system_load": {
      "cpu": 0.65,
      "memory": 0.78,
      "gpu": 0.82
    }
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid input parameters |
| `MODEL_NOT_FOUND` | Requested model does not exist |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Server-side error |
| `TIMEOUT_ERROR` | Request timeout |
| `RESOURCE_EXHAUSTED` | System resources unavailable |

## Rate Limits

| Endpoint Type | Limit | Window |
|---------------|-------|---------|
| Authentication | 100 requests | 1 hour |
| Predictions | 1000 requests | 1 hour |
| Training | 10 jobs | 1 day |
| General API | 10000 requests | 1 hour |

## SDK Examples

### Python SDK
```python
from pno_physics_bench import PNOClient

# Initialize client
client = PNOClient(api_key="your_api_key")

# Load model
model = client.get_model("pno-navier-stokes-v1")

# Make prediction with uncertainty
result = model.predict(
    initial_condition=[[1.0, 2.0, 3.0]],
    parameters={"reynolds_number": 100},
    num_samples=100,
    confidence_levels=[0.9, 0.95]
)

print(f"Mean prediction: {result.mean}")
print(f"Uncertainty: {result.uncertainty.total}")
print(f"90% CI: {result.confidence_intervals['90']}")
```

### JavaScript SDK
```javascript
import { PNOClient } from '@terragonlabs/pno-physics-bench';

// Initialize client
const client = new PNOClient({
  apiKey: 'your_api_key',
  baseURL: 'https://api.pno.terragonlabs.com/v1'
});

// Make prediction
const result = await client.predict('pno-navier-stokes-v1', {
  input: {
    initial_condition: [[1.0, 2.0, 3.0]],
    parameters: { reynolds_number: 100 }
  },
  options: {
    num_samples: 100,
    confidence_levels: [0.9, 0.95]
  }
});

console.log('Prediction:', result.prediction);
console.log('Uncertainty:', result.uncertainty);
```

## Webhooks

### Training Completion
When a training job completes, a webhook is sent to the configured endpoint:

```json
{
  "event": "training.completed",
  "job_id": "job_train_abc123",
  "model_id": "pno-custom-model",
  "status": "success",
  "final_metrics": {
    "rmse": 0.078,
    "nll": -2.45,
    "coverage_90": 91.2
  },
  "timestamp": "2025-08-20T23:22:56.673Z"
}
```

## Support

- **Documentation**: https://docs.pno.terragonlabs.com
- **API Reference**: https://api-docs.pno.terragonlabs.com
- **Support**: support@terragonlabs.com
- **Status Page**: https://status.pno.terragonlabs.com

---

*This API documentation is automatically generated and updated.*
