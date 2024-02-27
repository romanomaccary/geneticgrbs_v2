# GeneticGRBs

This repository contains the Python code that implements a Genetic Algorithm (GA) optimization of the seven parameters of the [Stern & Svensson (1996)](https://iopscience.iop.org/article/10.1086/310267) (SS96) stochastic pulse avalanche model for the generation of simulated GRB light curves, developed in [this]() Astronomy & Astrophysics (A&A) paper (also on [arXiv]()).

_Reference Paper_:

_Corresponding Author_: 

_Code Contributors_ (in alphabetical order): G. Angora, L. Bazzanini, L. Ferro, C. Guidorzi, and A. Tsvetkova.

The GA has been implemented using [`PyGAD`](https://github.com/ahmedfgad/GeneticAlgorithmPython) (`v3.3.1`), a FOSS Python library containing a collection of several machine learning algorithms.

The Python code that implements the SS96 stochastic model was orginally forked from [this public repository](https://github.com/anastasia-tsvetkova/lc_pulse_avalanche), but has undergone significant changes.

<p align="center">
<img src="avalanche.png"  alt="" width = "450" />
</p>

Please cite the following paper if you use this code in your work:
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



## Installation

Follow the instructions in the file `conda_pygad3_env.txt`, namely:

1. `conda create -n pygad3 python=3.10`

2. `conda activate pygad3`

3. `pip install numpy==1.26.4 matplotlib==3.8.3 seaborn==0.13.2 pandas==2.2.0 astropy==6.0.0 pygad==3.3.1 tqdm==4.66.2 scipy==1.12.0 cloudpickle==3.0.0 h5py==3.10.0 astroml==1.0.2.post1 scikit-learn==1.4.1.post1`

4. `conda install ipykernel --update-deps --force-reinstall`



## Usage
