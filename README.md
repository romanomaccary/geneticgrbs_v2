# GeneticGRBs

This repository contains the Python code that implements a Genetic Algorithm (GA) optimization of the seven parameters of the [Stern & Svensson (1996)](https://iopscience.iop.org/article/10.1086/310267) (SS96) stochastic pulse avalanche model for the generation of simulated GRB light curves, developed in [this]() Astronomy & Astrophysics (A&A) paper (also on [arXiv]()).

The GA has been implemented using [`PyGAD`](https://github.com/ahmedfgad/GeneticAlgorithmPython) (`v3.3.1`), a FOSS Python library containing a collection of several machine learning algorithms.

The Python code that implements the SS96 stochastic model was orginally forked from [this public repository](https://github.com/anastasia-tsvetkova/lc_pulse_avalanche), but has undergone significant changes.

<p align="center">
<img src="avalanche.png"  alt="" width = "450" />
</p>