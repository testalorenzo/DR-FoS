#
# Simulation results analysis
#

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
        samples[mask] = sample_truncated_normal(mean, std, 0.02, 0.98, np.sum(mask))
    return samples

def set_size(width, fraction=1, subplots=(3, 3)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

if __name__ == "__main__":

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=0.55)
    plt.rcParams["axes.grid"] = False
    # plt.rcParams['text.usetex'] = True
    width = 396

    version = '4'
    data = pd.read_csv('sim_results_DRFoS' + version + '.csv')
    # Round to 2 decimal places
    data = data.round(4)
    # Replace zeros with small values
    data['error_or'] = data['error_or'].replace(0, 1e-3)

    axd = plt.figure(figsize=set_size(width, subplots=(1,4))).subplot_mosaic(
        [['example', '.', 'misspi', 'missmu', '.', 'coverage']],
        width_ratios=[1, 0.15, 1, 1, 0.15, 1],
        sharey=False,
        gridspec_kw = {'wspace':0.05, 'hspace':0.05}
    )

    # MISSPECIFIED PROPENSITY SCORE
    print('Misspecified propensity score')
    bad_data = data[(data.alpha_mu == 0)]

    avg_error_ipw = bad_data.error_ipw.mean()
    avg_error_or = bad_data.error_or.mean()
    avg_error_drfos = bad_data.error_drfos.mean()
    avg_coverage = bad_data.coverage.mean()

    std_error_ipw = bad_data.error_ipw.std()
    std_error_or = bad_data.error_or.std()
    std_error_drfos = bad_data.error_drfos.std()
    std_coverage = bad_data.coverage.std()

    bad_data = bad_data.loc[:, ['alpha_pi', 'error_ipw', 'error_or', 'error_oracle', 'error_drfos']]#.melt('alpha_pi')

    x = bad_data.alpha_pi.unique()
    y = bad_data.error_ipw.groupby(bad_data.alpha_pi).mean()
    axd['misspi'].plot(x, y, label='IPW', color=sns.color_palette()[1], linewidth=1)

    # axd['misspi'].axhline(y=bad_data.error_oracle.mean(), color=sns.color_palette()[3], linestyle='--', label='Oracle', linewidth=1)
    y = bad_data.error_drfos.groupby(bad_data.alpha_pi).mean()
    axd['misspi'].plot(x, y, label='DR-FoS', color=sns.color_palette()[0], linewidth=1)
    # axd['misspi'].set_yscale('log')
    # add error bars
    axd['misspi'].set_xticks(ticks=[0,0.25,0.5,0.75,1], labels=['0', '.25', '.5', '.75', '1'])
    axd['misspi'].set_yticks(ticks=[0,0.25,0.5,0.75], labels=['0', '.25', '.5', '.75'])

    axd['misspi'].set_title('Robustness wrt $\\pi$')
    axd['misspi'].set_xlabel('$\\alpha_{\\pi}$')
    # axd['misspi'].set_ylabel('MSE')
    axd['misspi'].tick_params(pad=-3)


    # bad_data.variable = bad_data.variable.map({'error_drfos': 'DR-FoS', 'error_ipw': 'IPW', 'error_or': 'OR'})
    # bad_data.variable = pd.Categorical(bad_data.variable, categories=['DR-FoS', 'IPW', 'OR'])

    # # plt.figure()
    # # g = sns.boxplot(x='alpha_pi', y="value", hue="variable", data=bad_data, palette="pastel", showfliers=False, whis=1.5)
    # g = sns.boxplot(x='alpha_pi', y="value", hue="variable", data=bad_data, palette="pastel", showfliers=False, whis=1.5, ax=axd['misspi'], log_scale=True)
    # sns.move_legend(axd['misspi'], "lower center", bbox_to_anchor=(1.03, -1.1), ncol=3, title='Method', frameon=True)
    # # plt.show()
   
    # # plt.savefig('misspecified_pi' + version + '.pdf', bbox_inches='tight')

    print('Average IPW error: ', avg_error_ipw, 'std: ', std_error_ipw)
    print('Average OR error: ', avg_error_or, 'std: ', std_error_or)
    print('Average DRFoS error: ', avg_error_drfos, 'std: ', std_error_drfos)
    print('Average coverage: ', avg_coverage, 'std: ', std_coverage)

    # MISSPECIFIED REGRESSION MODEL
    print('Misspecified regression model')
    bad_data = data[(data.alpha_pi == 0)]

    avg_error_ipw = bad_data.error_ipw.mean()
    avg_error_or = bad_data.error_or.mean()
    avg_error_drfos = bad_data.error_drfos.mean()
    avg_coverage = bad_data.coverage.mean()

    std_error_ipw = bad_data.error_ipw.std()
    std_error_or = bad_data.error_or.std()
    std_error_drfos = bad_data.error_drfos.std()
    std_coverage = bad_data.coverage.std()

    bad_data = bad_data.loc[:, ['alpha_mu', 'error_ipw', 'error_or', 'error_oracle', 'error_drfos']]#.melt('alpha_mu')

    x = bad_data.alpha_mu.unique()
    y = bad_data.error_or.groupby(bad_data.alpha_mu).mean()
    axd['missmu'].plot(x, y, label='OR', color=sns.color_palette()[2], linewidth=1)
    # axd['missmu'].axhline(y=bad_data.error_oracle.mean(), color=sns.color_palette()[3], linestyle='--', label='Oracle', linewidth=1)
    y = bad_data.error_drfos.groupby(bad_data.alpha_mu).mean()
    axd['missmu'].plot(x, y, label='DR-FoS', color=sns.color_palette()[0], linewidth=1)
    # axd['missmu'].set_yscale('log')

    axd['missmu'].set_xticks(ticks=[0,0.25,0.5,0.75,1], labels=['0', '.25', '.5', '.75', '1'])
    axd['missmu'].set_yticks(ticks=[0,0.25,0.5,0.75], labels=['', '', '', ''])
    axd['missmu'].set_xlabel('$\\alpha_{\\mu}$')
    axd['missmu'].set_title('Robustness wrt $\\mu$')
    axd['missmu'].tick_params(pad=-3)

    # bad_data.variable = bad_data.variable.map({'error_drfos': 'DR-FoS', 'error_ipw': 'IPW', 'error_or': 'OR'})
    # bad_data.variable = pd.Categorical(bad_data.variable, categories=['DR-FoS', 'IPW', 'OR'])

    # # plt.figure()
    # g = sns.boxplot(x='alpha_mu', y="value", hue="variable", data=bad_data, palette="pastel", showfliers=False, whis=1.5, ax=axd['missmu'], legend=False, log_scale=True)
    # # g = sns.boxplot(x='alpha_mu', y="value", hue="variable", data=bad_data, palette="pastel", whis=1.5)
    # # plt.savefig('misspecified_mu' + version + '.pdf', bbox_inches='tight')

    print('Average IPW error: ', avg_error_ipw, 'std: ', std_error_ipw)
    print('Average OR error: ', avg_error_or, 'std: ', std_error_or)
    print('Average DRFoS error: ', avg_error_drfos, 'std: ', std_error_drfos)
    print('Average coverage: ', avg_coverage, 'std: ', std_coverage)

    # Coverage

    bad_data = data[(data.alpha_mu == 0)]
    x = bad_data.alpha_pi.unique()
    y = bad_data.coverage.groupby(bad_data.alpha_pi).mean()
    axd['coverage'].plot(x, y, label='$\\alpha_\\pi$', color=sns.color_palette()[5], linewidth=1)

    bad_data = data[(data.alpha_pi == 0)]
    x = bad_data.alpha_mu.unique()
    y = bad_data.coverage.groupby(bad_data.alpha_mu).mean()
    axd['coverage'].plot(x, y, label='$\\alpha_\\mu$', color=sns.color_palette()[9], linewidth=1)

    axd['coverage'].set_xticks(ticks=[0,0.25,0.5,0.75,1], labels=['0', '.25', '.5', '.75', '1'])
    axd['coverage'].set_yticks(ticks=[0.94, 0.95, 0.975, 1], labels=['', '.95', '.97', '1'])
    axd['coverage'].axhline(y=0.95, color=sns.color_palette()[7], linestyle='--', linewidth=1)
    axd['coverage'].set_title('Coverage $\\Delta$')
    axd['coverage'].set_xlabel('Miss. level')
    axd['coverage'].tick_params(pad=-3)

    axd['coverage'].legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3, title='Miss. type', frameon=True)

    # EXAMPLE
    n = 5000 # number of observations
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

    np.random.seed(1)
    a_pi = 0.75
    a_mu = 0.25

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

    axd['example'].plot(grid, ipw, label='IPW', color=sns.color_palette()[1], linewidth=1)
    axd['example'].plot(grid, outcome_regression, label='OR', color=sns.color_palette()[2], linewidth=1)
    axd['example'].plot(grid, beta, label='True', color=sns.color_palette()[4 ], linewidth=1)
    axd['example'].plot(grid, drfos, label='DR-FoS', color=sns.color_palette()[0], linewidth=1)
    axd['example'].fill_between(grid, lower_bound, upper_bound, color=sns.color_palette()[0], alpha=0.2)
    axd['example'].set_title('Example')
    axd['example'].set_xlabel('Time')
    axd['example'].set_xticks(ticks=[0,0.25,0.5,0.75,1], labels=['0', '.25', '.5', '.75', '1'])
    axd['example'].tick_params(pad=-3)

    handles, labels = axd['misspi'].get_legend_handles_labels()
    handles2, labels2 = axd['missmu'].get_legend_handles_labels()
    handles3, labels3 = axd['example'].get_legend_handles_labels()
    labels_final = ['IPW', 'OR', 'DR-FoS', 'True $\\beta$']#, 'Oracle']
    handles_final = [handles[0], handles2[0], handles[1], handles3[2]]#, handles[1]]
    axd['misspi'].legend(handles_final, labels_final, loc='upper center', bbox_to_anchor=(0.4, -0.4), ncol=4, title='Method', frameon=True)

    plt.savefig('sim_results' + version + '.pdf', bbox_inches='tight', pad_inches=0)