# Security Policy

## Supported Versions

Only the latest commit on `main` receives security fixes.

| Version | Supported |
|---|---|
| `main` | ✅ |
| older commits | ❌ |

## Reporting a Vulnerability

**Please do not open a public issue for security reports.**

Use [GitHub private vulnerability reporting](https://github.com/drintoul/extractor/security/advisories/new)
instead — it stays private until a fix is released.

Please include:

- A description of the issue and its potential impact
- Steps to reproduce or a proof-of-concept
- Affected component (pipeline code, compose stack, UI, …)

You'll get an acknowledgement as soon as possible; fixes are handled on a
best-effort basis for this personal project.

## Scope Notes

This stack is designed to run **self-hosted on a private network**. Some
configurations are intentional for that trust model and are not vulnerabilities:

- Firecrawl runs with `USE_DB_AUTHENTICATION=false` — its API is internal-only
  and never published to the host by design
- All inter-service traffic (ollama, mongo, redis, rabbitmq, postgres) stays on
  the private `extractor-internal` bridge network

**Please do report** issues like:

- Anything that would expose internal services to the host/network
  unintentionally
- Secrets committed to the repository or logged
- Credential/SSRF/injection issues in the pipeline or UI code
- Vulnerable dependency pins

## Using This Project Safely

- Never commit `.env` — it's gitignored; `.env.example` uses placeholders
- Don't publish `APP_PORT` to an untrusted network: the UI can trigger runs
  against arbitrary URLs (SSRF by design — it's a scraper)
- Respect target sites' `robots.txt` and terms of service
