# GitHub Workflows Setup Instructions

The autonomous SDLC implementation created comprehensive CI/CD workflows, but these need to be manually added to GitHub due to permission restrictions.

## Required GitHub Actions Workflow

Create the following file in your repository: `.github/workflows/ci-cd.yml`

```yaml
name: PNO Physics Bench CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  release:
    types: [published]

env:
  PYTHON_VERSION: "3.9"
  NODE_VERSION: "18"

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: pip install -e .[dev]
      
      - name: Run security scan
        run: pip install bandit && bandit -r src/ -f json -o security-report.json || true
      
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to DockerHub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            terragonlabs/pno-physics-bench:latest
            terragonlabs/pno-physics-bench:${{ github.sha }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          echo 'Deploying to staging environment...'
          kubectl apply -f deployment/staging/ --kubeconfig=${{ secrets.KUBECONFIG_STAGING }}

  deploy-production:
    needs: [build, deploy-staging]
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    environment: production
    steps:
      - name: Deploy to production
        run: |
          echo 'Deploying to production environment...'
          kubectl apply -f deployment/production/ --kubeconfig=${{ secrets.KUBECONFIG_PRODUCTION }}
```

## Required GitHub Secrets

Add the following secrets to your GitHub repository settings:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token
- `KUBECONFIG_STAGING`: Base64 encoded kubeconfig for staging
- `KUBECONFIG_PRODUCTION`: Base64 encoded kubeconfig for production

## Setup Instructions

1. Navigate to your repository on GitHub
2. Create the `.github/workflows/` directory
3. Add the `ci-cd.yml` file with the content above
4. Configure the required secrets in repository settings
5. The workflow will automatically trigger on the next push

This completes the CI/CD pipeline setup for the autonomous SDLC implementation.