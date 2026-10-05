# Security policy

## Supported versions

| Branch | Status |
| --- | --- |
| `main` | Supported. Default branch; V6 MVP plus current docs. |
| `r7/weather-reasoning` | Supported. Active R7 research line. |
| other branches | Unsupported. Treat as experiments. |

## Reporting a vulnerability

Do not open a public issue for a security report.

Email `blog@eswlnk.com` with:

- affected path, branch, and commit if known
- impact (data leak, secret exposure, supply-chain, CI abuse)
- a minimal reproduction that does not include live credentials

Please allow 7 days for an acknowledgement. Credentials, tokens, and private dataset manifests must be rotated by the reporter as well as the maintainer.

This repository vendors third-party baselines under `legacy_v531_full/baselines/external_sources/`. Report issues in those trees upstream unless the vulnerability is in how this repo invokes them.
