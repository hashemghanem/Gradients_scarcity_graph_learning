# Supervised learning of analysis-sparsity priors with automatic differentiation

This repo contains the python scripts that reproduce the figures in the published paper.

As we also provide the package we built to learn analysis-sparsity priors, solving a bilevel optimization with Automatic Differentiation, in the directory <./modules>, these scripts serve as a guiding example to help users applying our framework on other datasets.

## Dependencies
First install needed packages using Conda by running:
```
conda env create -f environment.yml
```
Then, activate the created environment called ```tv```:
```
conda activate tv
```

## Running a script:
To run the script that generates a figure in the published paper, execute the following in the command line:
```
python -u name_of_script.py
```
after replacing ```name_of_script ``` by the name of the according script from the three ones we provide here. To reproduce figure 1 for example :
```
python -u plot_fig1_varying_noise_amplitude.py
```
