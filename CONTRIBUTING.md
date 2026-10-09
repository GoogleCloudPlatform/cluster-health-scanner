# How to Contribute

We'd love to accept your patches and contributions to Cluster Health Scanner (CHS). There are just a few guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License Agreement (CLA). You (or your employer) retain the copyright to your contribution; this simply gives us permission to use and redistribute your contributions as part of the project. Head over to <https://cla.developers.google.com/> to see your current agreements on file or to sign a new one.

## Discuss Before Implementing

* **Major Changes**: For new features, architectural changes, or new scheduler integrations, open an Issue first to discuss your idea with maintainers.
* **Minor Changes**: Bug fixes, documentation updates, and small improvements can be submitted directly as PRs.

## Contribution Workflow

### 1. Target Branch
* All pull requests must be opened against the **`develop`** branch. PRs targeting `main` will be closed.
* Direct pushes to repository branches are disabled; contributors cannot push or merge changes directly.

### 2. Pull Request Lifecycle
1. **Fork and Branch**: Create a feature branch from `develop` in your fork.
2. **Local Validation**: Set up your environment and run the pre-commit checks and unit tests as described in [DEVELOPMENT.md](DEVELOPMENT.md). All checks must pass before you open a PR.
3. **Submit PR**: Open a PR targeting `develop`. GitHub Actions will verify CLA compliance and run automated checks.
4. **Review & Merge**: Maintainers (`@GoogleCloudPlatform/cluster-health-scanner-maintainers`) review and merge PRs once approved. Contributors cannot merge their own PRs.

## Test Requirements

* All new features and bug fixes must include unit tests under `tests/`.
* PRs must pass all tests and maintain existing coverage.

## Review Expectations & Etiquette

* Maintainers review PRs on a best-effort basis (typically within 1–2 weeks).
* Keep PRs small and focused on a single change.
* Address review comments promptly and keep your branch rebased on `develop`.

## Community Guidelines

This project follows Google's Open Source Community Guidelines: <https://opensource.google/conduct/>

## AI-Assisted Contributions

Submissions to this project may use coding assistance tools, including AI-generated code, as long as these submissions abide by the Contributor License Agreement's requirements. All contributions will be reviewed to ensure they meet project standards.
