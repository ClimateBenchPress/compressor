# ClimateBenchPress

This repository contains the main functionality for the ClimateBenchPress compression benchmark.

## Getting Started

This project uses the uv package manager to handle dependencies. If you don't already have it installed follow the instructions at <https://docs.astral.sh/uv/getting-started/installation/>.

Next, clone this repository and within the project directory install all the necessary dependencies with:
```bash
uv sync
uv pip install -e "."
```

### Downloading the Data

Make sure you have all the necessary data downloaded by following the instructions at <https://github.com/ClimateBenchPress/data-loader>.

## Citation

If you find this work useful, please consider citing the following paper:
```bibtex
@Article{reichelt2026climatebenchpress,
  AUTHOR = {Reichelt, T. and Tyree, J. and Kl\"ower, M. and Dueben, P. and Lawrence, B. N. and Baker, A. H. and Faghih-Naini, S. and Hoefler, T. and Stier, P.},
  TITLE = {ClimateBenchPress (v1.0): a benchmark for lossy compression of climate data},
  JOURNAL = {Geoscientific Model Development},
  VOLUME = {19},
  YEAR = {2026},
  NUMBER = {13},
  PAGES = {5933--5960},
  URL = {https://gmd.copernicus.org/articles/19/5933/2026/},
  DOI = {10.5194/gmd-19-5933-2026}
}
```

## Funding

ClimateBenchPress has been developed as part of [Embed2Scale](https://embed2scale.eu/) and [ESiWACE3](https://www.esiwace.eu/).

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054 and EU’s Horizon Europe program under grant agreement number 101131841. This work also received funding from [UK Research and Innovation (UKRI)](https://www.ukri.org/).
