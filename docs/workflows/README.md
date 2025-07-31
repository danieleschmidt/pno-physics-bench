# CI/CD Workflows for PNO Physics Bench

This directory contains GitHub Actions workflow templates and documentation for setting up continuous integration and deployment for the PNO Physics Bench repository.

## Quick Setup

1. Copy the workflow files from `templates/` to `.github/workflows/`
2. Configure the required secrets in your GitHub repository settings
3. Customize the workflows based on your specific needs

## Available Workflows

### Core Workflows

1. **`ci.yml`** - Main CI pipeline for testing and quality checks
2. **`security.yml`** - Security scanning and dependency auditing
3. **`release.yml`** - Automated releases and PyPI publishing
4. **`docs.yml`** - Documentation building and deployment

### Specialized Workflows

5. **`benchmarks.yml`** - ML model benchmarking and performance tracking
6. **`docker.yml`** - Container image building and publishing
7. **`dependabot-auto-merge.yml`** - Automated dependency updates

## Required Repository Secrets

Configure these in your GitHub repository settings under Settings > Secrets and variables > Actions:

### PyPI Publishing
- `PYPI_API_TOKEN` - PyPI API token for package publishing
- `TEST_PYPI_API_TOKEN` - Test PyPI token for testing releases

### Container Registry
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password or access token

### Optional Integrations
- `CODECOV_TOKEN` - Codecov integration token
- `SONAR_TOKEN` - SonarCloud integration token
- `WANDB_API_KEY` - Weights & Biases API key for ML experiment tracking

## Workflow Configuration

### Matrix Strategy

All workflows use matrix strategies for comprehensive testing:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ['3.9', '3.10', '3.11']
    pytorch-version: ['2.0.0', '2.1.0', 'latest']
```

### Caching Strategy

- Python dependencies cached using `actions/cache`
- PyTorch models and datasets cached for faster CI
- Docker layer caching for container builds

## Security Features

- SLSA Level 3 compliance for supply chain security
- Dependency vulnerability scanning with GitHub Security
- Code scanning with CodeQL
- Container image scanning with Trivy
- SBOM (Software Bill of Materials) generation

## Performance Benchmarking

The benchmark workflow automatically:
- Runs performance tests on representative datasets
- Compares results against baseline metrics
- Generates performance reports
- Fails CI if performance regression exceeds threshold

## Local Development

Test workflows locally using [act](https://github.com/nektos/act):

```bash
# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Test the CI workflow
act -W .github/workflows/ci.yml

# Test with specific event
act pull_request -W .github/workflows/ci.yml
```

## Monitoring and Alerts

Workflows are configured to:
- Send Slack notifications on failure (configure `SLACK_WEBHOOK`)
- Create GitHub issues for failed scheduled runs
- Generate detailed failure reports with logs and artifacts

## Workflow Triggers

Each workflow is optimized for appropriate triggers:

- **CI**: Pull requests, pushes to main/develop
- **Security**: Daily scheduled scans, dependency changes
- **Release**: Tag pushes (v*), manual dispatch
- **Docs**: Changes to docs/, README updates
- **Benchmarks**: Weekly scheduled runs, manual dispatch
- **Docker**: Releases, changes to containerization files

## Best Practices

1. **Branch Protection**: Configure branch protection rules requiring CI to pass
2. **Required Reviews**: Require at least 1 review for production changes
3. **Status Checks**: Make CI workflow checks required for merging
4. **Auto-merge**: Use dependabot auto-merge for minor updates
5. **Artifact Retention**: Configure appropriate retention periods for build artifacts

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **CUDA Issues**: Use CPU-only PyTorch for CI unless testing GPU functionality
3. **Memory Limits**: Reduce batch sizes or dataset sizes for CI
4. **Timeout Issues**: Split long tests or increase timeout values

### Performance Optimization

- Use faster runners (ubuntu-latest) for most jobs
- Parallel test execution with pytest-xdist
- Minimal dependency installation for different job types
- Conditional job execution based on file changes

## Integration Checklist

- [ ] Copy workflow templates to `.github/workflows/`
- [ ] Configure required repository secrets
- [ ] Update branch protection rules
- [ ] Test workflows with a test PR
- [ ] Configure notification channels
- [ ] Set up monitoring dashboards
- [ ] Document any custom configurations

For detailed implementation examples, see the `templates/` directory.