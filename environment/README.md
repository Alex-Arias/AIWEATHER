# AIWeather Development Environment

This directory contains the official development environment for AIWeather.

## Files

- `environment.yml` — Conda environment specification
- `requirements.txt` — Exact pip package versions

## Create the environment

```bash
conda env create -f environment.yml
conda activate aiweather
```

## Update after installing new packages

```bash
conda env export --no-builds > environment.yml
pip freeze > requirements.txt
```