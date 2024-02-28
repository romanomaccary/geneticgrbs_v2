# GeneticGRBs

This repository contains the Python code that implements a Genetic Algorithm (GA) for the optimization of the seven parameters of the [Stern & Svensson (1996)](https://iopscience.iop.org/article/10.1086/310267) (SS96) stochastic pulse avalanche model, for the generation of simulated gamma-ray burst (GRB) light curves (LCs), developed in [this paper]() ([arXiv version]()).

- _Reference Paper_: [Long gamma-ray burst light curves as the result of a common stochastic pulse-avalanche process]().

- _Corresponding Author_: Lorenzo Bazzanini, University of Ferrara ( `bzzlnz  at unife.it` )

- _Code Contributors_ (in alphabetical order): G. Angora, L. Bazzanini, L. Ferro, C. Guidorzi, and A. Tsvetkova.


## Description

In order to run the GA minimization procedure, the BATSE or the _Swift_/BAT light curves are needed. Instead, if you just want to simulate a new set of LCs (given a set of seven SS96 parameters) the code in this repository should be sufficient.

The GA has been implemented using [`PyGAD`](https://github.com/ahmedfgad/GeneticAlgorithmPython) ([Gad 2023](https://link.springer.com/article/10.1007/s11042-023-17167-y)), a FOSS Python library containing a collection of several machine learning algorithms.

The Python code that implements the SS96 stochastic model is also contained in this repo; it was originally forked from [this public repository](https://github.com/anastasia-tsvetkova/lc_pulse_avalanche) by one of the co-authors, but has undergone significant changes.

<p align="center">
<img src="avalanche.png"  alt="" width = "450" />
</p>



## Installation

To run the code, it is advised to create a self-contained `conda` environment with all the proper libraries installed. To create the aformentioned environment, called `pygad3`, just run the following four lines of code (or follow the instructions in the file `./conda_pygad3_env.txt`, which contains also the version of all installed packages):
```
# 1. create the conda environment 
conda create -n pygad3 python=3.10

# 2. activate the environment
conda activate pygad3

# 3. install all the required libraries (with the required version)
pip install numpy==1.26.4 matplotlib==3.8.3 seaborn==0.13.2 pandas==2.2.0 astropy==6.0.0 pygad==3.3.1 tqdm==4.66.2 scipy==1.12.0 cloudpickle==3.0.0 h5py==3.10.0 astroml==1.0.2.post1 scikit-learn==1.4.1.post1

# 4. install some extra utilities for VSCode
conda install ipykernel --update-deps --force-reinstall
```


## Usage

### Generation of a set of simulated LCs
The value of the optimized parameters (see paper) are:
```
# BATSE
mu      = 1.02
mu0     = 0.96
alpha   = 2.84
delta1  = -1.32
delta2  = 0.28
tau_min = 0.02
tau_max = 34.8

# Swift/BAT
mu      = 
mu0     = 
alpha   = 
delta1  = 
delta2  = 
tau_min = 
tau_max = 

```

To generate the LCs, change the value of the parameters in the file `./simulate_GRBs.py `, and then run:
```
# move to the right directory
cd ./lc_pulse_avalanche
# activate the conda env
conda activate pygad3
# simulate a set of LCs
python3 simulate_GRBs.py
```

### Running the GA optimization
```
# move to the right directory
cd ./genetic_algorithm
# activate the conda env
conda activate pygad3
# run the GA
python3 testGA.py
```



## Other
If you have any question or you are interested in contributing do not hesitate to contact the authors.

Please cite the following associated paper if you use this code in your work:
```
@article{,
  title={},
  author={},
  journal={},
  pages={},
  year={},
  publisher={}
}
```
