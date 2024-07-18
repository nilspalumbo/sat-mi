# Mechanistically Interpreting a Transformer-based 2-SAT Solver: An Axiomatic Approach

## Installing Dependencies

First, make sure conda is installed. In the base environment, run:

```
conda install nb_conda_kernels
conda env create -f environment.yml
conda activate 2sat_interp
npm install vega-lite vega-cli canvas --global
```

Next, build [drat-trim](https://github.com/marijnheule/drat-trim):

```
git clone https://github.com/marijnheule/drat-trim.git
cd drat-trim
make
cp drat-trim ..
```

## Generating the training and analysis datasets

Use the notebook `RandomCNFs.ipynb`; see "Generate Training Dataset" and "Generate Analysis Dataset." Datasets used for experiments are available [here](https://zenodo.org/records/11239102); download these to the "data" folder. The dataset used for training and testing the model is named cnf_tokens_1M.npy and the one used for our analysis is named cnf_tokens_100K.npy.

## Training the model

`python -u model.py --num_heads 1 4 --num_layers 2 --run_name layers_2_heads_1_4 --data_path ./data/cnf_tokens_1M.npy --train`

The trained model used for analysis and a log of the training process are in the "models/layers_2_heads_1_4" folder.

## Code structure

`RandomCNFs.ipynb` contains the code to generate the datasets, `model.py` contains the model implementation, `plot.py` contains plotting code, `helpers.py` contains various utilities. The core code for the analysis of the modl is in `interpretation.ipynb` and with the key implementation details in `interpretation.py`, including the decomposition of the model, our interpretation, and the alpha and gamma functions.
`model.py` and `plot.py` are adapated from [Progress measures for grokking via mechanistic interpretability](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper).

## Performing mechanistic interpretability analysis

The notebook `interpretation.ipynb` contains the code to run the analyis.

