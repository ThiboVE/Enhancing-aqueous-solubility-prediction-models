# Enhancing-aqueous-solubility-prediction-models-with-quantum-descriptors
Internship project at QSimulate of @ThiboVE

## Introduction


This project aims to compare the predictive power of quantum descriptors against traditional topological descriptors in cheminformatics workflows. Specifically, we will use standard machine learning techniques to train machine learning models for optimally predicting the aqueous solubility of organic compounds. The project will be executed in three stages, the final of which is an open-ended research phase.

**Stage 1**: Dataset curation and feature generation with QuantumFP

**Stage 2**: Model training and recursive feature elimination

**Stage 3**: Further enhancement of models through feature engineering


## Code setup

To access the code used in this project follow the next steps:

I have used a UV virtual environment, so make sure UV is installed.

1. Activate the UV environment, in terminal: `.venv\Scripts\activate`
2. Install the packages in `requirements.txt` with `uv pip install -r requirements.txt`
3. install my custom (local) package with `uv pip install -e src/`