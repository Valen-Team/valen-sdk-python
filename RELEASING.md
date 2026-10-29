# Publishing Python SDK

`valen-sdk==0.2.0` was published to [PyPI](https://pypi.org/project/valen-sdk/0.2.0/) on 2026-10-26 using the account-configured Trusted Publisher. Build, wheel/sdist metadata checks and OIDC publication passed in [the workflow](https://github.com/Valen-Team/valen-sdk-python/actions/runs/40286898341). PyPI files, SHA256, installed wheel tests and source equivalence are recorded in `validation/pypi-0.2.0.json`. GitHub distribution hashes are recorded separately because archive metadata can differ between builds.

## Configuration

PyPI publisher: project `valen-sdk`, GitHub owner `Valen-Team`, repository `valen-sdk-python`, workflow `publish.yml`, environment `pypi`. The GitHub environment exists. A pending publisher creates the project on its first successful publication. No long-lived PyPI token is required; do not reuse GitHub/HF tokens for the registry.

The workflow is manually dispatched and currently builds the pinned `v0.3.0` tag. For a future version, update package versions/changelog, run appropriate tests, create its reviewed tag, then update the checkout ref in `.github/workflows/publish.yml` before dispatch. Never overwrite an existing PyPI version or move a published tag. Keep the workflow name and environment aligned with PyPI, and configure environment reviewer protection as appropriate.

## Verification

Query `https://pypi.org/pypi/valen-sdk/VERSION/json`; check the actual version, file inventory and SHA256. Install the small wheel in isolation with model extras disabled, then run contract tests. Reuse cached model assets for real inference checks. Update the README, release notes and website only after registry publication is verified.

The JavaScript SDK is independently published on npm as `@valen-team/sdk@0.2.0`; see the JavaScript repository's RELEASING.md.
