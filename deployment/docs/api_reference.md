# PNO Physics Bench - API Reference

## Model Inference API

### POST /predict
Perform model inference with uncertainty quantification.

**Request Body:**
```json
{
  "input": "tensor data as nested array",
  "num_samples": 10,
  "return_uncertainty": true
}
```

**Response:**
```json
{
  "prediction": "tensor data",
  "uncertainty": "uncertainty tensor",
  "inference_time": 0.045,
  "model_info": {
    "version": "1.0.0",
    "parameters": 1250000
  }
}
```

## Training API

### POST /train
Start model training with specified configuration.

### GET /training/status
Get current training status and metrics.

### POST /training/stop
Stop current training job.

## Management API

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-08T20:00:00Z",
  "checks": {
    "model_loaded": true,
    "gpu_available": false,
    "memory_usage": 0.65
  }
}
```

### GET /metrics
Prometheus metrics endpoint.

### GET /info
System information.

**Response:**
```json
{
  "version": "1.0.0",
  "pytorch_version": "2.1.0",
  "cuda_version": null,
  "model_info": {
    "type": "ProbabilisticNeuralOperator",
    "parameters": 1250000,
    "last_updated": "2025-08-08T20:00:00Z"
  }
}
```

## Error Codes

- `400` - Bad Request (invalid input)
- `422` - Unprocessable Entity (model error)
- `500` - Internal Server Error
- `503` - Service Unavailable (model not loaded)
