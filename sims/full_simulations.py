#
# Additional simulation results
#

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

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
    fig_height_in = 1.5 * fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

if __name__ == "__main__":

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=0.55)
    plt.rcParams["axes.grid"] = False
    # plt.rcParams['text.usetex'] = True
    width = 396

    version = '5'
    data = pd.read_csv('sim_results_DRFoS' + version + '.csv')
    # Round to 2 decimal places
    data = data.round(4)
    # Replace zeros with small values
    data['error_or'] = data['error_or'].replace(0, 1e-3)

    list_of_axes = []
    for a_pi, a_mu in itertools.product([0, 0.25, 0.50, 0.75, 1], [0, 0.25, 0.50, 0.75, 1]):
        list_of_axes.append('alpha' + str(a_mu) + str(a_pi))
    list_of_axes = [list_of_axes[i:i + 5] for i in range(0, len(list_of_axes), 5)]


    axd = plt.figure(figsize=set_size(width, subplots=(5,5))).subplot_mosaic(
        list_of_axes,
        width_ratios=[1, 1, 1, 1, 1],
        sharey=True,
        sharex=True,
        gridspec_kw = {'wspace':0.05, 'hspace':0.05}
    )

    for a_pi, a_mu in itertools.product([0, 0.25, 0.50, 0.75, 1], [0, 0.25, 0.50, 0.75, 1]):
                
        bad_data = data[(data.alpha_mu == a_mu) & (data.alpha_pi == a_pi)].loc[:,['error_drfos', 'error_ipw', 'error_or']]
        bad_data.columns = ['DR-FoS', 'IPW', 'OR']
        bad_data = bad_data.melt()

        ax = axd['alpha' + str(a_mu) + str(a_pi)]
        sns.boxplot(x='variable', y="value", data=bad_data, palette="pastel", showfliers=False, whis=1.5, ax=ax)

        if a_mu == 0:
            ax.set_ylabel('$\\alpha_{\\pi} = ' + str(a_pi) + '$')
        if a_pi == 0:
            ax.set_title('$\\alpha_{\\mu} = ' + str(a_mu) + '$')
        ax.set_xlabel('')
        #ax.set_yticks([])
        ax.tick_params(pad=-3)

    plt.savefig('sim_results_DRFoS' + version + '_boxplot.pdf', bbox_inches='tight', pad_inches=0)