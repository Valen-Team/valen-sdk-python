## Prepared release 0.4.0

Prepared on 2026-10-31. Publication is pending; verify the registry before updating stable install instructions. See CHANGELOG.md for compatibility notes.

# Publishing Python SDK

`valen-sdk==0.3.1` was published to [PyPI](https://pypi.org/project/valen-sdk/0.3.1/) on 2026-10-29 from tag `v0.3.1` ([workflow](https://github.com/Valen-Team/valen-sdk-python/actions/runs/40903656187)). `valen-sdk==0.3.0` was published to [PyPI](https://pypi.org/project/valen-sdk/0.3.0/) on 2026-10-29 ([workflow](https://github.com/Valen-Team/valen-sdk-python/actions/runs/40844916532), `validation/pypi-0.3.0.json`). `valen-sdk==0.2.0` was published to [PyPI](https://pypi.org/project/valen-sdk/0.2.0/) on 2026-10-26 using the account-configured Trusted Publisher. Build, wheel/sdist metadata checks and OIDC publication passed in [the workflow](https://github.com/Valen-Team/valen-sdk-python/actions/runs/40286898341). PyPI files, SHA256, installed wheel tests and source equivalence are recorded in `validation/pypi-0.2.0.json`. GitHub distribution hashes are recorded separately because archive metadata can differ between builds.

## Configuration

PyPI publisher: project `valen-sdk`, GitHub owner `Valen-Team`, repository `valen-sdk-python`, workflow `publish.yml`, environment `pypi`. The GitHub environment exists. A pending publisher creates the project on its first successful publication. No long-lived PyPI token is required; do not reuse GitHub/HF tokens for the registry.

Pushing a `vX.Y.Z` tag runs `.github/workflows/publish.yml`, which builds that tag and publishes it. For a release, set the version in `pyproject.toml` and `src/valen_sdk/__init__.py`, date the changelog, run the tests, then tag the reviewed commit and push the tag. Never overwrite an existing PyPI version or move a published tag. Keep the workflow name and environment aligned with PyPI, and configure environment reviewer protection as appropriate.

## Verification

Query `https://pypi.org/pypi/valen-sdk/VERSION/json`; check the actual version, file inventory and SHA256. Install the small wheel in isolation with model extras disabled, then run contract tests. Reuse cached model assets for real inference checks. Update the README, release notes and website only after registry publication is verified.

The JavaScript SDK is independently published on npm as `@valen-team/sdk@0.2.0`; see the JavaScript repository's RELEASING.md.
