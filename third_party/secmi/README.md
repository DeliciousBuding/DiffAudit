# Vendored SecMI Subset

This directory contains a minimal subset of the official SecMI repository that is needed for adapter integration and local experimentation.

Source repository:

- https://github.com/jinhaoduan/SecMI

Upstream license:

- MIT, retained in [LICENSE](LICENSE)

Included files are limited to the attack-side modules and model definition that are needed to study or integrate the official SecMI attack path:

- `model.py`
- `mia_evals/secmia.py`
- `mia_evals/dataset_utils.py`
- `mia_evals/resnet.py`
- `mia_evals/member_splits/*`

The full upstream repository is not committed inside this project. Local exploratory clones can be placed under `external/`, which is ignored by git.
