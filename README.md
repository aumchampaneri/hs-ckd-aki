# Mapping complement cascade dynamics in chronic kidney disease (CKD) and acute kidney injury (AKI).

## Abstract
[Insert]

---

## Environment
For maximum reproducibility, we recommend using the included pixi environment. To install pixi, follow the instructions [here](https://pixi.prefix.dev/latest/).

## Hardware
The analysis was performed on a 16" M1 Max MacBook Pro with 64GB RAM. Scripts have not been validated for use on other hardware.

## Script organization
These scripts are organized into stages, with each stage corresponding to a different processing step.

#### Data acquisition
- `01a_download.py`
- `01b_metrics.py`

#### Latent space analysis
- `02a_scvi-processing.py`
- `02b_scvi-plot.py`

#### Pseudobulk differential expression
- `03a_pseudobulk-de.py`
- `03b_pydeseq2-plot.py`
