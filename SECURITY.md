# Security Policy

## Supported Versions

The latest published crate version and the latest tagged release are supported with security updates. Older versions are not maintained separately.

| Version | Supported |
|---------|-----------|
| 0.4.x   | ✅ Yes |
| < 0.4   | ❌ No  |

## Reporting a Vulnerability

Please do **not** open a public issue for security vulnerabilities.

Instead, report privately via email to:

**Eugene Khodzitsky** — `eugene.khodzitsky@gmail.com`

Include the following information:
1. A clear description of the vulnerability.
2. Steps to reproduce (minimal proof of concept if possible).
3. The affected version(s) and platform(s).
4. Your assessment of severity and impact.

You can expect an initial response within **7 days**. If the report is accepted, we will work on a fix and coordinate disclosure. Credit will be given unless you prefer to remain anonymous.

## Security Measures

- SHA-256 verification of all downloaded model files.
- Sanitized error responses (no path or internal model leakage to clients).
- Origin allowlist and CORS controls.
- Per-IP rate limiting (opt-in).
- Connection, frame, and body size limits.
- Graceful shutdown with bounded drain timeouts.
- Weekly Dependabot scans and `cargo audit` in CI.

## Past Advisories

None at this time.
