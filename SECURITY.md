# Security Policy

## Supported Versions

We actively support security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it to:

- **Email**: security@terragonlabs.com
- **Security Advisory**: Create a private security advisory on GitHub

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if known)

### Response Timeline

- **Initial Response**: Within 24 hours
- **Status Update**: Weekly until resolved
- **Fix Timeline**: Critical issues within 72 hours, others within 30 days

## Security Measures

This project implements the following security measures:

1. **Input Validation**: All user inputs are validated and sanitized
2. **Secure Subprocess**: Use of `subprocess.run()` instead of `subprocess.call()`
3. **Parameterized Queries**: No string formatting in SQL queries
4. **No Hardcoded Secrets**: All secrets managed through environment variables
5. **Security Scanning**: Automated security scans on all pull requests
6. **Dependency Scanning**: Regular scanning of dependencies for vulnerabilities

## Security Best Practices

- Never commit secrets or API keys to the repository
- Use environment variables for sensitive configuration
- Validate all inputs from external sources
- Use parameterized queries for database operations
- Keep dependencies up to date
- Follow principle of least privilege

## Automated Security

This project includes automated security measures:

- Pre-commit hooks for secret detection
- Dependency vulnerability scanning
- Static code analysis for security issues
- Runtime security monitoring

## Security Contact

For security-related questions or concerns:
- Email: security@terragonlabs.com
- PGP Key: Available upon request
