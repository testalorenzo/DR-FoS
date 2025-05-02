import numpy as np
import pyreadr
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.linear_model import LogisticRegression, LinearRegression
from fungcn.fgcn.solver_fgcn import FunGCN
from torch import device

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

    # Make torch run on cpu
    dvc = device("cpu")

    seed = 1998
    np.random.seed(seed)

    withcomp = False

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=0.55)
    plt.rcParams["axes.grid"] = False

    width = 396

    axd = plt.figure(figsize=set_size(width, subplots=(1,4))).subplot_mosaic(
        [['main160', 'main170', '.', 'main169', 'main179']],
        width_ratios=[1, 1, 0.15, 1, 1],
        gridspec_kw = {'wspace':0.05, 'hspace':0.05}
    )

    # FunGCN parameters
    forecast_ratio = 0.
    pruning = 0.7
    k_gcn = 10
    lr = 5e-5
    max_selected = 5
    k_graph = 3
    nhid = [32, 32]
    epochs = 50
    batch_size = 1
    dropout = 0.
    kernel_size = 0
    patience = 5
    min_delta = 0
    val_size = 0
    test_size = 0
    save_graph_name = None

    # Import Rdata
    result = pyreadr.read_r('data_for_fasten.Rdata') # Available on request
    result.keys()

    data = result['Amat'].to_numpy()
    description = result['descriptions_var']
    types = result['type_var']

    # Model for regression
    regressor = 'FunGCN' # 'Linear'

    # Effect of hypertension on mobility index adjusted for covariates
    idx_treatments = [16, 17]
    idx_responses = [0, 9]
    idx_covariates = [26, 27, 29, 34, 35, 36]

    # for loop over all combinations
    for idx_treat, idx_response in itertools.product(idx_treatments, idx_responses):
        
        good_subset = data[idx_treat,:,:].sum(axis=1) == 192 # 419
        control = data[idx_treat,:,:].sum(axis=1) == 0 # 577

        Y_treatment = data[idx_response,good_subset,:]
        Y_control = data[idx_response,control,:]

        X_treatment = data[:,good_subset,0]
        X_control = data[:,control,0]
        X_treatment = X_treatment[idx_covariates,:].T
        X_control = X_control[idx_covariates,:].T

        X = np.concatenate((X_treatment, X_control), axis=0)
        Y = np.concatenate((Y_treatment, Y_control), axis=0)
        A = np.concatenate((np.ones(sum(good_subset)), np.zeros(sum(control))), axis=0)
        n = X.shape[0]
        p = X.shape[1]

        # Preprocess data
        if regressor == 'FunGCN':
            X_repeated = np.repeat(X[:, :, np.newaxis], 192, axis=2).transpose(1, 0, 2)
            B = np.concat([Y.reshape(1, n, 192), X_repeated], axis=0)
            modalities = types.to_numpy()[[idx_response] + idx_covariates].reshape(-1).tolist()

        if withcomp:
            # Fit logistic regression model
            model = LogisticRegression(max_iter=100000)
            model.fit(X, A)
            pi_hat = model.predict_proba(X)[:,1]

            # Fit function-on-scalar regression model
            if regressor == 'Linear':
                model_control = LinearRegression()
                model_control.fit(X_control, Y_control)

                model_treatment = LinearRegression()
                model_treatment.fit(X_treatment, Y_treatment)

                Y_hat_control = model_control.predict(X)
                Y_hat_treatment = model_treatment.predict(X)
            else:
                model_control = FunGCN(data=B[:, A==0, :], y_ind=[0], var_modality=modalities, verbose=1)
                model_treatment = FunGCN(data=B[:, A==1, :], y_ind=[0], var_modality=modalities, verbose=1)
                # preprocess the data
                model_control.preprocess_data(k_graph=k_graph, k_gcn=k_gcn, forecast_ratio=forecast_ratio,
                                        test_size=test_size, val_size=val_size, random_state=seed)
                model_treatment.preprocess_data(k_graph=k_graph, k_gcn=k_gcn, forecast_ratio=forecast_ratio,
                                        test_size=test_size, val_size=val_size, random_state=seed)
                # create graph
                model_control.graph_estimation(max_selected=max_selected, graph_path=save_graph_name)
                model_treatment.graph_estimation(max_selected=max_selected, graph_path=save_graph_name)
                # initialize GCN model
                model_control.initialize_gcn_model(pruning=pruning, nhid=nhid, dropout=dropout, kernel_size=kernel_size)
                model_treatment.initialize_gcn_model(pruning=pruning, nhid=nhid, dropout=dropout, kernel_size=kernel_size)
                # train GCN model
                model_control.train_model(lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, min_delta=min_delta)
                model_treatment.train_model(lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, min_delta=min_delta)
                # predict on test set
                Y_hat_control = model_control.predict(new_data=B).y_hat_os.reshape(n, 192)
                Y_hat_treatment = model_treatment.predict(new_data=B).y_hat_os.reshape(n, 192)

            outcome_regression = np.mean(Y_hat_treatment - Y_hat_control, axis=0)
            ipw = np.mean(A[:,None] * Y / pi_hat[:,None] - (1 - A[:,None]) * Y / (1 - pi_hat[:,None]), axis=0)
            
        # Cross validation with balanced treatment and control groups
        n_folds = 5
        n_treatment = sum(good_subset)
        n_control = sum(control)

        # Split the data into n_folds
        idx_treatment = np.arange(n_treatment)
        idx_control = np.arange(n_control)

        np.random.shuffle(idx_treatment)
        np.random.shuffle(idx_control)

        idx_treatment = np.array_split(idx_treatment, n_folds)
        idx_control = np.array_split(idx_control, n_folds)

        drfos = np.zeros(192)
        cov_dr_fos = np.zeros((n,192))
        cov_dr_fos_opt2 = np.zeros((1,192))
    
        # Loop over the folds
        for idx_fold in range(n_folds):
            
            idx_treatment_train = np.concatenate([idx_treatment[i] for i in range(n_folds) if i != idx_fold])
            idx_control_train = np.concatenate([idx_control[i] for i in range(n_folds) if i != idx_fold])

            idx_treatment_test = idx_treatment[idx_fold]
            idx_control_test = idx_control[idx_fold]

            Y_treatment_train = Y_treatment[idx_treatment_train,:]
            Y_control_train = Y_control[idx_control_train,:]

            X_treatment_train = X_treatment[idx_treatment_train,:]
            X_control_train = X_control[idx_control_train,:]

            Y_treatment_test = Y_treatment[idx_treatment_test,:]
            Y_control_test = Y_control[idx_control_test,:]

            X_treatment_test = X_treatment[idx_treatment_test,:]
            X_control_test = X_control[idx_control_test,:]

            # Concatenate the training sets
            Y_train = np.concatenate((Y_treatment_train, Y_control_train), axis=0)
            X_train = np.concatenate((X_treatment_train, X_control_train), axis=0)
            A_train = np.concatenate((np.ones(n_treatment - len(idx_treatment_test)), np.zeros(n_control - len(idx_control_test))), axis=0)
            
            Y_test = np.concatenate((Y_treatment_test, Y_control_test), axis=0)
            X_test = np.concatenate((X_treatment_test, X_control_test), axis=0)
            A_test = np.concatenate((np.ones(len(idx_treatment_test)), np.zeros(len(idx_control_test))), axis=0)

            n_train = X_train.shape[0]
            n_test = X_test.shape[0]

            if regressor == 'FunGCN':
                X_train_repeated = np.repeat(X_train[:, :, np.newaxis], 192, axis=2).transpose(1, 0, 2)
                B_train = np.concat([Y_train.reshape(1, n_train, 192), X_train_repeated], axis=0)
                X_test_repeated = np.repeat(X_test[:, :, np.newaxis], 192, axis=2).transpose(1, 0, 2)
                B_test = np.concat([Y_test.reshape(1, n_test, 192), X_test_repeated], axis=0)
                modalities = types.to_numpy()[[idx_response] + idx_covariates].reshape(-1).tolist()

            # Fit logistic regression model
            model = LogisticRegression(max_iter=100000)
            model.fit(X_train, A_train)
            pi_hat = model.predict_proba(X_test)[:,1]

            # Fit function-on-scalar regression model
            if regressor == 'Linear':
                model_control = LinearRegression()
                model_control.fit(X_control_train, Y_control_train)

                model_treatment = LinearRegression()
                model_treatment.fit(X_treatment_train, Y_treatment_train)

                Y_hat_control = model_control.predict(X_test)
                Y_hat_treatment = model_treatment.predict(X_test)
            else:
                model_control = FunGCN(data=B_train[:, A_train==0, :], y_ind=[0], var_modality=modalities, verbose=1)
                model_treatment = FunGCN(data=B_train[:, A_train==1, :], y_ind=[0], var_modality=modalities, verbose=1)
                # preprocess the data
                model_control.preprocess_data(k_graph=k_graph, k_gcn=k_gcn, forecast_ratio=forecast_ratio,
                                        test_size=test_size, val_size=val_size, random_state=seed)
                model_treatment.preprocess_data(k_graph=k_graph, k_gcn=k_gcn, forecast_ratio=forecast_ratio,
                                        test_size=test_size, val_size=val_size, random_state=seed)
                # create graph
                model_control.graph_estimation(max_selected=max_selected, graph_path=save_graph_name)
                model_treatment.graph_estimation(max_selected=max_selected, graph_path=save_graph_name)
                # initialize GCN model
                model_control.initialize_gcn_model(pruning=pruning, nhid=nhid, dropout=dropout, kernel_size=kernel_size)
                model_treatment.initialize_gcn_model(pruning=pruning, nhid=nhid, dropout=dropout, kernel_size=kernel_size)
                # train GCN model
                model_control.train_model(lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, min_delta=min_delta)
                model_treatment.train_model(lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, min_delta=min_delta)
                # predict on test set
                Y_hat_control = model_control.predict(new_data=B_test).y_hat_os.reshape(n_test, 192)
                Y_hat_treatment = model_treatment.predict(new_data=B_test).y_hat_os.reshape(n_test, 192)

            # Compute the average treatment effect
            drfos_fold = np.mean(Y_hat_treatment + A_test[:,None] * (Y_test - Y_hat_treatment) / pi_hat[:,None] - Y_hat_control - (1 - A_test[:,None]) * (Y_test - Y_hat_control) / (1 - pi_hat[:,None]), axis=0)

            drfos += drfos_fold

            cov_dr_fos_opt_fold = Y_hat_treatment + A_test[:,None] * (Y_test - Y_hat_treatment) / pi_hat[:,None] - Y_hat_control - (1 - A_test[:,None]) * (Y_test - Y_hat_control) / (1 - pi_hat[:,None]) # - drfos_fold
            cov_dr_fos_opt2 = np.concatenate([cov_dr_fos_opt2, cov_dr_fos_opt_fold], axis=0)

            # Compute covariance
            if regressor == 'Linear':
                Y_hat_treatment_full = model_treatment.predict(X)
                Y_hat_control_full = model_control.predict(X)
            else:
                Y_hat_control_full = model_control.predict(new_data=B).y_hat_os.reshape(n, 192)
                Y_hat_treatment_full = model_treatment.predict(new_data=B).y_hat_os.reshape(n, 192)

            pi_hat_full = model.predict_proba(X)[:,1]
        
            cov_dr_fos_fold = Y_hat_treatment_full + A[:,None] * (Y - Y_hat_treatment_full) / pi_hat_full[:,None] - Y_hat_control_full - (1 - A[:,None]) * (Y - Y_hat_control_full) / (1 - pi_hat_full[:,None]) - drfos_fold
            cov_dr_fos += cov_dr_fos_fold

        drfos /= n_folds
        cov_dr_fos /= n_folds
        cov_dr_fos2 = cov_dr_fos.T @ cov_dr_fos
        cov_dr_fos2 /= n
        cov_dr_fos3 = np.cov(cov_dr_fos, rowvar=False)
        
        cov_dr_fos_opt2 = cov_dr_fos_opt2[1:,:]
        cov_dr_fos_opt3 = cov_dr_fos_opt2.T @ cov_dr_fos_opt2 / n**2
        cov_dr_fos_opt3 = np.cov(cov_dr_fos_opt2, rowvar=False) / n

        # Simulate 95% confidence bands from Gaussian Process centerd in drfos and with covariance matrix cov_dr_fos
        n_sim = 10000
        drfos_sim = np.random.multivariate_normal(drfos, cov_dr_fos_opt3, n_sim)
        lower_bound = np.quantile(drfos_sim, 0.025, axis=0)
        upper_bound = np.quantile(drfos_sim, 0.975, axis=0)

        # Plot the results
        axd['main' + str(idx_treat) + str(idx_response)].plot(range(192), drfos, label='DR-FoS', linewidth=1)
        axd['main' + str(idx_treat) + str(idx_response)].fill_between(range(192), lower_bound, upper_bound, alpha=0.35)
        axd['main' + str(idx_treat) + str(idx_response)].set_xlabel('Time (months)')
        if idx_response == 0 and idx_treat == 16:
            axd['main' + str(idx_treat) + str(idx_response)].set_ylabel('Estimated FATE')
        if idx_response == 0 and idx_treat == 16:
            axd['main' + str(idx_treat) + str(idx_response)].set_ylim(-2.3, 0.1)
            axd['main' + str(idx_treat) + str(idx_response)].set_yticks([-2, -1, 0])
        if idx_response == 9 and idx_treat == 16:
            axd['main' + str(idx_treat) + str(idx_response)].set_ylim(-0.02, 0.4)
            axd['main' + str(idx_treat) + str(idx_response)].set_yticks(ticks=[0, 0.2, 0.4], labels=['0', '.2', '.4'])
        if idx_response == 0 and idx_treat == 17:
            axd['main' + str(idx_treat) + str(idx_response)].set_ylim(-2.3, 0.1)
            axd['main' + str(idx_treat) + str(idx_response)].set_yticks(ticks=[-2, -1, 0], labels=['','',''])
        if idx_response == 9 and idx_treat == 17:
            axd['main' + str(idx_treat) + str(idx_response)].set_ylim(-0.02, 0.4)
            axd['main' + str(idx_treat) + str(idx_response)].set_yticks(ticks=[0, 0.2, 0.4], labels=['','',''])

        axd['main' + str(idx_treat) + str(idx_response)].axhline(y=0, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)

        if withcomp:
            axd['main' + str(idx_treat) + str(idx_response)].plot(range(192), outcome_regression, label='OR', linewidth=1)
            axd['main' + str(idx_treat) + str(idx_response)].plot(range(192), ipw, label='IPW', linewidth=1)

        axd['main' + str(idx_treat) + str(idx_response)].set_xlim(0, 192)
       
        axd['main' + str(idx_treat) + str(idx_response)].set_xticks([0, 40, 80, 120, 160])
        axd['main' + str(idx_treat) + str(idx_response)].tick_params(pad=-3)

        axd['main' + str(idx_treat) + str(idx_response)].set_title('Effect of ' + description['descriptions_var'][idx_treat] + '\n on ' + description['descriptions_var'][idx_response])
    
    if withcomp:
        plt.savefig('DR-FoS_estimates' + str(idx_response) + '_' + str(idx_treat) + '_' + regressor + '_withcomp.pdf', bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('DR-FoS_estimates' + str(idx_response) + '_' + str(idx_treat) + '_' + regressor + '.pdf', bbox_inches='tight', pad_inches=0)