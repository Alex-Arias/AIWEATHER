# AIWeather Development Environment

This directory contains the **official reproducible development environment** for AIWeather.

The goal is to ensure that every developer (or future version of ourselves) can recreate the exact software environment used to develop and test AIWeather.

---

## Contents

| File                    | Purpose                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| `environment.yml`       | Conda environment specification (recommended for recreating the environment).                |
| `requirements.txt`      | Exact Python packages installed with `pip freeze`. Useful for debugging and reproducibility. |
| `export_environment.sh` | Helper script to regenerate the environment files after package updates.                     |
| `README.md`             | Documentation for managing the development environment.                                      |

---

# Creating the environment

Create a fresh environment from the exported Conda specification:

```bash
conda env create -f environment.yml
conda activate aiweather
```

---

# Updating the environment

Whenever new packages are installed or existing packages are upgraded, regenerate the environment files:

```bash
conda env export --no-builds > environment.yml
pip freeze > requirements.txt
```

If `export_environment.sh` is available:

```bash
./export_environment.sh
```

After verifying the changes:

```bash
git add environment/
git commit -m "Update AIWeather development environment"
git push
```

---

# Development policy

The `aiweather` Conda environment is considered the **stable development environment** for the project.

Before introducing major dependency upgrades (for example, a new Earth2Studio release), create a separate experimental environment such as:

```text
aiweather-dev
earth2studio17
```

Only after the new environment has been fully tested should the official development environment be updated.

This policy minimizes downtime and prevents breaking the primary development environment.

---

# Versioning

The exported environment should always correspond to the current stable version of AIWeather.

For example:

* AIWeather **v0.1.0** → `environment.yml` reflects the environment used to produce the first working GraphCast backend.

Keeping the environment synchronized with project releases ensures long-term reproducibility and makes it possible to reproduce past experiments exactly.
