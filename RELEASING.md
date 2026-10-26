# Releasing 0.2.0

Publication status: GitHub artifact release; PyPI/npm not yet published. Package names were absent from their registries at preflight. No registry credentials, repository publishing secrets or existing trusted-publishing workflows were found. GitHub/HF credentials must not be reused for package registries.

Verify the GitHub release SHA256SUMS before installation. Tests cover HTTP contracts, errors, retries, assets, lifecycle and preprocessing. Existing real-model validation is reused; package smoke tests do not establish new backend/platform support.

Python: configure a PyPI Trusted Publisher for Valen-Team/valen-sdk-python, workflow `publish.yml`, environment `pypi`, or use a scoped PyPI token outside Git. Run the manual workflow after configuring the publisher. The environment must exist and should require maintainer review. The workflow builds and checks wheel/sdist, then uses OIDC; it does not run on ordinary push.

JavaScript: an authorized npm account must own/create @valen scope and have publish rights for @valen/sdk. From a clean v0.2.0 checkout run `npm test`, `npm run typecheck`, `npm pack --dry-run`, then `npm publish --access public` with the account's authorized authentication/2FA. No npm credential is stored in this repository. Optionally configure npm trusted publishing in npm after establishing package ownership.

After publication query `https://pypi.org/pypi/valen-sdk/0.2.0/json` and `https://registry.npmjs.org/@valen%2fsdk/0.2.0`, compare filenames/digests, perform a clean small-package install with optional model dependencies disabled, and update README and release notes to reflect actual registry availability.
