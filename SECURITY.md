Security Policy
===============

This is a small, local-only neural network library written in the Goose
language. It is an educational/demo project with no network access, no
authentication, no cryptography, and no handling of user data or secrets.

## Supported Versions

Only the current `master` branch is maintained. No versioned releases exist.

## Reporting Vulnerabilities

There is no dedicated security contact or bounty program. If you find a
security issue in the Goose runtime or compiler used by this project
(`GOOSE_BINARY`), report it to the Goose project
maintainers, not here.

## Scope

This project:

- Makes no outbound network requests
- Reads/writes only files you explicitly point it at (e.g. `mat_save` /
  `mat_load` to paths you supply)
- Does not authenticate, authorize, or manage users
- Does not implement or use any cryptographic primitives
- Does not process personal data

Because it runs entirely locally and reads/writes only caller-supplied files,
the attack surface is limited to the Goose compiler/runtime itself. Treat
compiled binaries (`src/run/main`, `src/code/test/main`) as untrusted
artifacts — they are gitignored and not shipped.

## Disclaimer

Provided as-is, for learning purposes. No warranty of security or fitness
for any particular purpose. Do not use for production systems.
