# AIWeather Architecture v1.0

**Version:** 1.0

**Status:** Architecture Freeze

---

# Overview

AIWeather is a modular AI forecasting framework designed to support multiple
meteorological and oceanographic AI models through a common preprocessing
pipeline and standardized internal data representation.

The framework is intentionally model-independent.

Supported AI models may include:

- GraphCast
- AIFS
- FourCastNet
- Pangu
- FuXi
- StormCast
- Future ocean AI models

Likewise, multiple data sources can be used:

- ERA5
- GFS
- IFS
- HRRR
- WRF
- GLORYS
- HYCOM
- CROCO
- ROMS

The objective is to decouple

    Data Sources

from

    AI Models

through a standardized preprocessing pipeline.

---

# High-Level Architecture

                Raw Data
                    │
                    ▼
             Dataset Adapter
                    │
                    ▼
           Standard Dataset
                    │
                    ▼
        Common Preprocessing
                    │
                    ▼
          Model Adapter Layer
                    │
                    ▼
          Earth2Studio Models
                    │
                    ▼
               AI Forecast

---

# Repository Structure

aiweather/

    datasets/

    preprocessing/

    model_adapters/

    models/

    forecast/

    verification/

    visualization/

---

# Package Responsibilities

## datasets/

Responsible for reading external datasets.

Examples:

- ERA5
- GFS
- IFS
- WRF
- HYCOM

This layer never knows anything about AI models.

---

## preprocessing/

Contains utilities shared by every model.

Responsibilities:

- validation
- coordinate handling
- latitude sorting
- longitude conversion
- tensor generation
- CoordSystem generation
- normalization

This layer never calls Earth2Studio.

---

## model_adapters/

Converts a Standard Dataset into the exact format required
by a specific AI model.

Examples:

GraphCastAdapter

FourCastNetAdapter

AIFSAdapter

Each adapter is model-specific.

---

## models/

Thin wrappers around Earth2Studio.

Responsibilities:

- loading checkpoints
- running inference
- returning forecasts

No preprocessing logic should exist here.

---

## forecast/

High-level forecasting workflows.

Example:

forecast()

hindcast()

ensemble()

---

# Design Principles

## 1. Separation of Concerns

Each package has one responsibility.

---

## 2. Model Independence

Models never know where data originated.

---

## 3. Dataset Independence

Datasets never know which AI model will consume them.

---

## 4. Reusable Preprocessing

All coordinate utilities and tensor generation belong
to the preprocessing layer.

---

## 5. Thin Model Wrappers

Model wrappers should contain as little logic as possible.

---

# Future Roadmap

Phase 1

✔ GraphCast

Phase 2

- ERA5 Adapter

Phase 3

- AIFS

Phase 4

- FourCastNet

Phase 5

- Pangu

Phase 6

- Ocean AI models

---

# Long-Term Vision

AIWeather aims to become a unified framework for
AI-based weather and ocean forecasting capable of
supporting multiple datasets, multiple AI models,
and multiple forecast workflows through a common,
extensible architecture.