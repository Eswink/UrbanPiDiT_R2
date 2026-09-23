# Install active R7 code (#27)

The distribution name/version remain unchanged for compatibility; this is not
a tagged release or PyPI publication. The wheel contains active data/preprocess,
data/download, model/layers and training modules, not raw data or legacy trees.

```bash
pip install '.[data,test]'
urbanpidit-r7-train --help
urbanpidit-r7-evaluate --help
```

CLI configuration files remain user-provided or available in the source repo;
there is no hidden dataset download. The real-data converters need the optional
`data` extra. Other production downloaders may additionally need the existing
requirements-data.txt dependencies and user-approved network access.

CI builds an actual wheel in a temporary source copy with --no-build-isolation
--no-index --no-deps, inspects its members, installs it to an isolated target and
runs import/CLI checks from a clean working directory. System dependencies remain
available for the test; project imports must come from the installed target.
This verifies installation on the CI platform, not every OS/GPU configuration.
