#
# Plot explicit simulation results
#

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

if __name__ == '__main__':

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=0.55)
    plt.rcParams["axes.grid"] = False
    # plt.rcParams['text.usetex'] = True
    width = 396

    axd = plt.figure(figsize=set_size(width, subplots=(1,4))).subplot_mosaic(
        [['main', 'fun']],
        gridspec_kw = {'wspace':0.1, 'hspace':0.05}
    )

    data = pd.read_csv('sim_results_DRFoS_explicit.csv')
    data['IPW_gcn'] = data['IPW']
    data['IPW_m_gcn'] = data['IPW_m']

    # Split in two depending on name of col.
    data_gcn = data.filter(like='gcn')
    data_lr = data.loc[:,~data.columns.str.contains('gcn')]

    # Rename columns
    data_gcn.columns = data_gcn.columns.str.replace('_gcn', '')

    # Reorder columns for plotting
    data_gcn = data_gcn.loc[:,data_lr.columns.tolist()]

    # Rename columns
    names = ['OR', 'IPW', 'DR', 'OR\n$\mu_m^{(a)}$', 'IPW\n$\pi_m^{(1)}$', 'DR\n$\pi_m^{(1)}$', 'DR\n$\mu_m^{(a)}$', 'DR\n$\pi_m^{(1)},\mu_m^{(a)}$']
    data_gcn.columns = names
    data_lr.columns = names

    data_gcn = data_gcn.melt(var_name='variable', value_name='value')
    data_lr = data_lr.melt(var_name='variable', value_name='value')

    sns.boxplot(x='variable', y="value", data=data_lr, palette="pastel", showfliers=True, whis=1.5, ax=axd['main'], fliersize=3)
    
    axd['main'].set_xlabel('Estimator')
    axd['main'].set_ylabel('Estimation error')
    axd['main'].set_title('Function-on-scalar linear regression')
    axd['main'].tick_params(pad=-3)

    sns.boxplot(x='variable', y="value", data=data_gcn, palette="pastel", showfliers=True, whis=1.5, ax=axd['fun'], fliersize=3)
    axd['fun'].set_xlabel('Estimator')
    axd['fun'].set_ylabel('')
    axd['fun'].set_title('FunGCN')
    axd['fun'].tick_params(pad=-3)

    plt.savefig('sim_results_DRFoS_explicit_boxplot.pdf', bbox_inches='tight', pad_inches=0)