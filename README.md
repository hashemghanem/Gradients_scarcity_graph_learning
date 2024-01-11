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
To run the script one needs to pass a few arguments:

- **dataset** from: Cora, CiteSeer, PubMed, Cheaters, Synthetic1HighFrequency, Synthetic1LowFrequency.
- **inner_model** from: GNN_simple, APPNP, Laplacian.
- **method** the version of the bilevel framework from: BO, BO+G2G, BO+regularization.
- **plot** from: True, False. A boolean deciding whether the code is run to only plot the hypergradient at iteration 9 or for a full optimization of the bilevel model. In the former case, only 9 outer iterations are performed.

For instance, to plot the hypergradient obtained at the outer iteration 9 on Cora, while using the 2-layer GNN in the paper, and using the bilevel framework without any fix (BO), one can execute:
```
python -u learn.py --dataset Cora --inner_model GNN_simple --method BO --plot True
```

Other hyperparameters can also be passed as optional arguments. To get a list of these options, execute the following command while still activating ```GradientScarcity``` environment:
```
python  learn.py -h
```