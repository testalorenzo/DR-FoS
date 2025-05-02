#
# Simulation analysis -- main
#

import numpy as np
import seaborn as sns
import pandas as pd
import csv
import itertools
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import Matern
from scipy.stats import truncnorm

# Define a function to sample from a truncated normal
def sample_truncated_normal(mean, std, lower, upper, size):
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

# Function to sample from the mixture
def sample_mixture(n_samples):
    means = [0.2, 0.8]  # Means of the truncated Gaussians
    stds = [0.1, 0.1]   # Standard deviations
    weights = [0.5, 0.5]  # Mixing weights (must sum to 1)
    # Randomly assign samples to components based on weights
    components = np.random.choice(len(weights), size=n_samples, p=weights)
    
    # Generate samples from the corresponding component
    samples = np.zeros(n_samples)
    for i, (mean, std) in enumerate(zip(means, stds)):
        mask = components == i
        samples[mask] = sample_truncated_normal(mean, std, 0.01, 0.99, np.sum(mask))
    
    return samples

if __name__ == "__main__":

    sns.set_theme(style="whitegrid", palette="pastel")

    print('Initializing simulations...')
    version = '5'
 
    # results storer
    columns=['seed', 'n', 'alpha_mu', 'alpha_pi', 'error_ipw', 'error_or', 'error_oracle', 'error_drfos', 'coverage']
    results = pd.DataFrame(columns=columns)

    with open('sim_results_DRFoS' + version + '.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow(columns)
    
    # Simulation parameters
    seed_of_seeds = 6
    number_of_seeds = 50
    seeds = np.random.RandomState(seed_of_seeds).choice(range(0, 1000), number_of_seeds, replace=False).tolist() # simulation seeds

    n = 50 # number of observations
    alpha_mu = np.arange(0,1.1,0.25) # number of features to remove from regression model
    alpha_pi = np.arange(0,1.1,0.25) # number of features to remove from propensity score model

    t = 100 # number of time points
    snr = 10 # signal-to-noise ratio

    n_sim = 10000 # Number of simulations for GP confidence bands

    sd_beta = 1  # standard deviation of the Matern covariance
    l_beta = 0.25  # range parameter of Matern covariance
    nu_beta = 3.5  # smoothness of Matern covariance

    sd_rho = 1  # standard deviation of the Matern covariance
    l_rho = 0.25  # range parameter of Matern covariance
    nu_rho = 3.5  # smoothness of Matern covariance

    l_eps = 0.25  # range parameter of eps Matern covariance
    nu_eps = 2.5  # smoothness of eps Matern covariance

    sd_U = 1  # standard deviation of the Matern covariance
    l_U = 0.25  # range parameter of Matern covariance
    nu_U = 2.5  # smoothness of Matern covariance

    domain = np.array([0, 1])  # domains of the curves
    grid = np.linspace(domain[0], domain[1], t)

    for itr in itertools.product(seeds, alpha_mu, alpha_pi):

        seed = itr[0]
        a_mu = itr[1]
        a_pi = itr[2]

        print('Seed:', seed, 'Error mu:', a_mu, 'Error pi:', a_pi)

        #
        # Generate data
        #

        np.random.seed(seed)

        # Create a random treatment assignment
        p_tilde = np.ones(n) / 2
        A = np.random.binomial(1, p_tilde, n)

        # Create rho, baseline function
        cov_rho = sd_rho ** 2 * Matern(length_scale=l_rho, nu=nu_rho)(grid.reshape(-1, 1))
        rho = np.random.multivariate_normal(np.zeros(t), cov_rho, 1)
        rho = rho[0]

        # Sample beta, causal effect
        cov_beta = sd_beta ** 2 * Matern(length_scale=l_beta, nu=nu_beta)(grid.reshape(-1, 1))
        beta = np.random.multivariate_normal(np.zeros(t), cov_beta, 1)
        beta = beta[0]

        # Compute regression functions
        mu1 = np.ones(n)[:,None] * beta + np.ones(n)[:,None] * rho
        mu0 = np.ones(n)[:,None] * rho

        # Compute response
        Y = A[:,None] * mu1 + (1-A[:,None]) * mu0

        # Add noise
        sd_eps = np.std(Y) / np.sqrt(snr)
        cov_eps = sd_eps ** 2 * Matern(length_scale=l_eps, nu=nu_eps)(grid.reshape(-1, 1))
        eps = np.random.multivariate_normal(np.zeros(t), cov_eps, n)
        eps -= eps.mean(axis=0)

        Y += eps

        #
        # Fit models
        #

        # Fit IPW
        U = sample_mixture(n) # sample_truncated_normal(0.5, 1, 0.02, .98, n) # np.random.uniform(0.02, 0.98, n) #
        pi_hat = (1 - a_pi) * p_tilde + a_pi * U
        w1 = A / pi_hat
        w0 = (1-A) / (1-pi_hat)
        w = w1 - w0
        ipw = np.mean(w[:,None] * Y, axis=0)

        # Fit OR
        cov_U = sd_U ** 2 * Matern(length_scale=l_U, nu=nu_U)(grid.reshape(-1, 1))
        U = np.random.multivariate_normal(np.zeros(t), cov_beta, n)

        mu_hat_treatment = (1 - a_mu) * mu1 + a_mu * U
        mu_hat_control = (1 - a_mu) * mu0 + a_mu * U
        outcome_regression = np.mean(mu_hat_treatment - mu_hat_control, axis=0)

        # Fit oracle (knows true propensity score and regression model)
        oracle = np.mean(mu1 + A[:,None] * (Y - mu1) / p_tilde[:,None] - mu0 - (1 - A[:,None]) * (Y - mu0) / (1 - p_tilde[:,None]), axis=0)

        # Fit DR-FoS
        drfos = np.mean(mu_hat_treatment + A[:,None] * (Y - mu_hat_treatment) / pi_hat[:,None] - mu_hat_control - (1 - A[:,None]) * (Y - mu_hat_control) / (1 - pi_hat[:,None]), axis=0)
        drfos_cov = np.cov(mu_hat_treatment + A[:,None] * (Y - mu_hat_treatment) / pi_hat[:,None] - mu_hat_control - (1 - A[:,None]) * (Y - mu_hat_control) / (1 - pi_hat[:,None]), rowvar=False) / n

        # Simulate 95% confidence bands from Gaussian Process centerd in drfos and with covariance matrix cov_dr_fos
        drfos_sim = np.random.multivariate_normal(drfos, drfos_cov, n_sim)
        lower_bound = np.quantile(drfos_sim, 0.025, axis=0)
        upper_bound = np.quantile(drfos_sim, 0.975, axis=0)

        # Compute error
        error_ipw = np.mean(np.abs((beta - ipw)))
        error_or = np.mean(np.abs((beta - outcome_regression)))
        error_oracle = np.mean(np.abs((beta - oracle)))
        error_drfos = np.mean(np.abs((beta - drfos)))
        # print(f"IPW error: {error_ipw}")
        # print(f"OR error: {error_or}")
        # print(f"DR-FoS error: {error_drfos}")

        # Compute coverage
        coverage = np.mean((beta >= lower_bound) & (beta <= upper_bound))
        # print(f"Coverage: {coverage}")

        # Store results
        series_to_export = pd.Series([seed, n, a_mu, a_pi, error_ipw, error_or, error_oracle, error_drfos, coverage])
        series_to_export.index = columns

        with open('sim_results_DRFoS' + version + '.csv','a') as fd:
            writer = csv.DictWriter(fd, fieldnames=columns)
            writer.writerow(series_to_export.to_dict())