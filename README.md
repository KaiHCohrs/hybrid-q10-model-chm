# Hybrid Q10 Model with Double Machine Learning

Hybrid modeling approaches for Q10 estimation from the ["Causal hybrid modeling with double machine learning—applications in carbon flux modeling"](https://iopscience.iop.org/article/10.1088/2632-2153/ad5a60) manuscript.

## Data Source

The model requires FLUXNET2015 half-hourly Fullset data. You can download this data from the [FLUXNET2015 Dataset](https://fluxnet.org/data/fluxnet2015-dataset/). After downloading, store the CSV files in the `data/` directory.

## Installation

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate hybrid-q10-model
```

2. Install the package:
```bash
pip install -e .
```

## Usage

To perform flux partitioning with the hybrid Q10 model:

1. Run either of the two scripts:
```bash
python scripts/DMLHM_Q10.py
```

or

```bash
python scripts/GDHM_Q10.py
```

for the DML-based or the GD-based Q10 estimation, respectively.

Note that the GD-based model is fully implemented in jax and hence runs in parallel while the 100 runs of the DML-based approach runs the 100 iterations sequentially.

## License

This project is licensed under the terms included in the LICENSE file. 