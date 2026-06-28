# %% Download data from CZ CELLxGENE
# > Download data from CELLxGENE using the CELLxGENE CLI and aquire citation
#
# %% PATH SETUP
from pathlib import Path

SCRIPT_DIR = Path.cwd()
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "outputs/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% IMPORTS
import cellxgene_census

# %%
census = cellxgene_census.open_soma(census_version="latest")
census["census_info"]["summary"].read().concat().to_pandas()
datasets = census["census_info"]["datasets"].read().concat().to_pandas()
# %%
dataset_id = "7ff0197b-d175-49bf-b4fa-150fe0995d93"
datasets[datasets["dataset_id"] == dataset_id].iloc[0]

file_name = f"{dataset_id}.h5ad"
file_path = DATA_DIR / file_name
# %%
if not file_path.exists():
    print(f"Downloading {dataset_id} to {file_path}...")
    cellxgene_census.download_source_h5ad(
        dataset_id, to_path=str(file_path), census_version="latest", progress_bar=True
    )
else:
    print(f"File already exists at {file_path}. Skipping download.")

census.close()
