# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅ |
| < 0.1   | ❌ |

## Reporting a Vulnerability

We take the security of PNO Physics Bench seriously. If you believe you have found a security vulnerability, please report it to us through coordinated disclosure.

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Instead, please send an email to daniel@terragonlabs.com with the following information:

- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### Response Timeline

We will respond to your report within **5 business days** with:
- Confirmation that we received your report
- Our assessment of the issue
- Expected timeline for a fix (if applicable)

We aim to resolve critical security issues within **30 days** of confirmation.

## Security Considerations for ML Research

### Data Security

- **Dataset Privacy**: Ensure training data doesn't contain sensitive information
- **Model Extraction**: Be aware that trained models can leak information about training data
- **Input Validation**: Always validate inputs to prevent adversarial attacks
- **Dependency Security**: Regularly update dependencies to patch known vulnerabilities

### Research Ethics

- **Dual Use**: Consider potential misuse of uncertainty quantification in safety-critical systems
- **Bias and Fairness**: Ensure uncertainty estimates don't amplify existing biases
- **Transparency**: Document limitations and failure modes of probabilistic predictions

### Deployment Security

- **Model Serving**: Secure model endpoints against unauthorized access
- **Resource Limits**: Implement appropriate resource limits to prevent DoS attacks
- **Logging**: Log security-relevant events without exposing sensitive data
- **Access Control**: Implement proper authentication and authorization

## Security Best Practices

### For Contributors

1. **Code Review**: All code must be reviewed before merging
2. **Dependency Scanning**: Use tools like `bandit` and `safety` to scan for vulnerabilities
3. **Secrets Management**: Never commit API keys, passwords, or other secrets
4. **Input Sanitization**: Validate and sanitize all user inputs
5. **Error Handling**: Don't expose sensitive information in error messages

### For Users

1. **Virtual Environments**: Always use isolated Python environments
2. **Regular Updates**: Keep the library and dependencies updated
3. **Resource Monitoring**: Monitor resource usage when running experiments
4. **Data Handling**: Follow your organization's data governance policies
5. **Network Security**: Use secure connections when downloading datasets

## Automated Security Measures

### Pre-commit Hooks

We use several automated security checks:

```yaml
# .pre-commit-config.yaml includes:
- repo: https://github.com/PyCQA/bandit
  rev: 1.7.4
  hooks:
    - id: bandit
      args: [-r, src/, -f, json, -o, bandit-report.json]
```

### Dependency Scanning

```bash
# Check for known vulnerabilities in dependencies
pip install safety
safety check

# Check for outdated packages
pip list --outdated
```

### Continuous Integration

Our CI pipeline includes:

- **Static Analysis**: Bandit security linting
- **Dependency Scanning**: Safety checks for known vulnerabilities
- **Secrets Detection**: Prevent accidental secret commits
- **License Scanning**: Ensure compatible licenses

## Vulnerability Disclosure Policy

### Scope

This policy applies to vulnerabilities in:

- PNO Physics Bench source code
- Documentation that could lead to security issues
- Build and deployment scripts
- Official Docker containers and configurations

### Out of Scope

- Third-party dependencies (report to upstream maintainers)
- Social engineering attacks
- Physical attacks against infrastructure
- Vulnerabilities in user's custom code or configurations

### Recognition

We maintain a security hall of fame to recognize researchers who help improve our security:

<!-- Security Hall of Fame -->
<!-- Thank you to the following researchers for responsibly disclosing security issues: -->
<!-- (This section will be populated as needed) -->

## Security Updates

Security updates will be announced through:

- **GitHub Security Advisories**: For critical vulnerabilities
- **Release Notes**: For all security-related fixes
- **Documentation**: Updated security guidance as needed
- **Mailing List**: If we establish one for security announcements

## Compliance and Standards

### Research Compliance

- **IRB Requirements**: Users must comply with institutional review board requirements for human subjects research
- **Data Protection**: Comply with GDPR, CCPA, and other applicable data protection regulations
- **Export Controls**: Be aware of export control regulations for cryptographic or dual-use technologies

### Industry Standards

- **NIST Cybersecurity Framework**: We align with NIST guidelines where applicable  
- **OWASP Top 10**: We consider OWASP recommendations for web-facing components
- **Supply Chain Security**: We follow SLSA guidelines for software supply chain security

## Contact Information

- **Security Team**: daniel@terragonlabs.com
- **PGP Key**: Available upon request for sensitive communications
- **Response Hours**: Monday-Friday, 9 AM - 5 PM PST

## Additional Resources

- [OWASP Machine Learning Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Adversarial ML Threat Matrix](https://github.com/mitre/advmlthreatmatrix)
- [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)

---

*This security policy is reviewed and updated regularly. Last updated: 2025-01-30*