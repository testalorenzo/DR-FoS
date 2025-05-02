# Doubly-Robust Functional Average Treatment Effect Estimation

This repository contains the code for the paper "Doubly-Robust Functional Average Treatment Effect Estimation" by Testa, Boschi, Chiaromonte, Kennedy and Reimherr (2025+) available on [arXiv](https://arxiv.org/abs/2501.06024).

The Python scripts in this repository implement the simulation study and the real-data analysis described in the paper. The files are organized as follows:

- */sims/sim_main.py*: This script implements the simulation study as described in the main text of our manuscript.
- */sims/explicit_simulations.py*: This script implements the explicit simulation study as described in the Supplementary Material of our paper.
- */sims/sim_results.py*: This script generates Figure 1, showing results for the simulation study in the main manuscript.
- */sims/full_simulations.py*: This script generates Figure D1 in the Supplementary Material.
- */sims/plot_explicit_simulations.py*: This script generates Figure D2 in the Supplementary Material.
- *app_analysis.py*: This script implements the real-data analysis on SHARE data. Data can be downloaded from the [SHARE website](https://share-eric.eu/data/). This script generates Figures 2 and E3.
