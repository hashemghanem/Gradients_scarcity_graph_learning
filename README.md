# Gradient scarcity with bilevel optimization for graph learning

This repo contains the python scripts that reproduce the figures in the published paper.

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

## Running a script:
To run the script that generates a figure in the published paper, execute the following in the command line:
```
python -u name_of_script.py
```
