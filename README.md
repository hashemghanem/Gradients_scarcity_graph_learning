# Gradient scarcity with bilevel optimization for graph learning

This repo contains the python scripts used to produce the figures in the according paper.

As we also provide the package we built to detect and resolve gradient scarcity in the directory <./modules>, these scripts serve as a guiding example to help users applying this package on other datasets.

## Dependencies
First install needed packages using Conda by running:
```
conda env create -f environment.yml
```
Then, activate the created environment called ```GradientScarcity```:
```
conda activate GradientScarcity
```

## Running the code:
The main script is learn.py.
To run the script one needs to specify:

- **the dataset** from: Cora, CiteSeer, PubMed, Cheaters, Synthetic1HighFrequency, Synthetic1LowFrequency.
- **the inner model** from: GNN_simple, APPNP, Laplacian.
- **the version of the bilevel framework** from: BO, BO+G2G, BO+regularization.

For instance, to experiment on Cora using the 2-layer GNN in the paper, and using the bilevel framework without any fix (BO), one can execute:
```
python -u learn.py Cora GNN_simple BO
```

Other hyperparameters can also be passed as optional arguments. To get a list of these options, execute the following command while still activating ```GradientScarcity``` environment:
```
python  learn.py -h
```