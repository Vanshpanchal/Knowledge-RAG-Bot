# GitHub Automation Templates

This repository keeps GitHub Actions as optional templates only.
No workflow runs automatically until a user copies templates into `.github/workflows/`.

## What Is Included

- `.github/workflow-templates/ci-template.yml`
- `.github/workflow-templates/docker-template.yml`
- `.github/workflow-templates/README.md`

These are safe starter templates for CI and Docker publishing.

## How To Enable A Workflow

1. Create the workflows folder if needed:
	- `mkdir -p .github/workflows`
2. Copy one or both templates:
	- `cp .github/workflow-templates/ci-template.yml .github/workflows/ci.yml`
	- `cp .github/workflow-templates/docker-template.yml .github/workflows/docker.yml`
3. Open the copied files and update placeholders.
4. Commit and push.

After copying, GitHub Actions will start running for the selected workflow.

## Required Placeholders To Update

In Docker template:

- `YOUR_DOCKERHUB_USERNAME`

In both templates:

- Branch filters (`main`, `develop`) if your branch strategy is different.

## Optional Secrets

For Docker Hub publish support, add repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

If these are not set, Docker Hub push steps are skipped.

## Notes

- Keeping templates in `.github/workflow-templates/` means no accidental CI/CD costs.
- The repository stays deployment-ready, but activation remains user-controlled.
- [Dependabot Configuration](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuring-dependabot-version-updates)
- [Docker Build Action](https://github.com/docker/build-push-action)
- [Codecov Integration](https://codecov.io)

---

## ✅ Checklist for First Run

- [ ] Repository pushed to GitHub
- [ ] Go to **Actions** tab (should show workflow runs)
- [ ] Check **docker.yml** completed successfully
- [ ] Check **ci.yml** completed successfully
- [ ] Image appears in **Packages** tab
- [ ] (Optional) Add `DOCKERHUB_TOKEN` if using Docker Hub
- [ ] (Optional) Enable "Auto-merge" for Dependabot PRs in **Settings**
