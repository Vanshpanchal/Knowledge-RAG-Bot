# Workflow Templates Guide

This folder contains optional GitHub Actions templates.
Templates in this directory do not run automatically.

## Available Templates

- `ci-template.yml`: lint, type-check, test, and security scan
- `docker-template.yml`: Docker build and publish (GHCR + optional Docker Hub)

## Enable A Template

Copy the template into `.github/workflows/`.

Example:

```bash
mkdir -p .github/workflows
cp .github/workflow-templates/ci-template.yml .github/workflows/ci.yml
cp .github/workflow-templates/docker-template.yml .github/workflows/docker.yml
```

## Update Required Values

In `docker-template.yml`, replace:

- `YOUR_DOCKERHUB_USERNAME`

If you use different default branches, update `on.push.branches` and `on.pull_request.branches`.

## Optional Repository Secrets

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Without those secrets, Docker Hub publishing steps are skipped.

## Disable Workflows

To disable any workflow, delete it from `.github/workflows/`.
