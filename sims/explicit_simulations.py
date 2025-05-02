#
# Explicit simulations
# 

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.gaussian_process.kernels import Matern
from fungcn.fgcn.solver_fgcn import FunGCN
from torch import device

if __name__ == "__main__":

    # Make torch run on cpu
    dvc = device("cpu") 

    seed_of_seeds = 6
    number_of_seeds = 50
    seeds = np.random.RandomState(seed_of_seeds).choice(range(0, 1000), number_of_seeds, replace=False).tolist() # simulation seeds

    or_errors = []
    ipw_errors = []
    dr_errors = []

    or_errors_gcn = []
    dr_errors_gcn = []

    or_errors_m = []
    ipw_errors_m = []
    dr_errors_mipw = []
    dr_errors_mor = []
    dr_errors_mboth = []

    or_errors_m_gcn = []
    dr_errors_mipw_gcn = []
    dr_errors_mor_gcn = []
    dr_errors_mboth_gcn = []

    for seed in seeds:

        np.random.seed(seed)

        # Parameters
        n = 1000
        p = 5
        t = 100 # number of time points

        domain = np.array([0, 1])  # domains of the curves
        grid = np.linspace(domain[0], domain[1], t)

        # Treatment effect
        D = np.ones(t) * 1

        # Simulate covariates
        X = np.random.uniform(-1, 1, (n, p))

        # Propensity score
        eta = np.random.normal(0, 1, p)
        logit = X @ eta + np.random.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-logit))
        A = np.random.binomial(1, ps)

        # Generate potential outcomes
        sd_beta = 1  # standard deviation of the Matern covariance
        l_beta = 0.25  # range parameter of Matern covariance
        nu_beta = 3.5  # smoothness of Matern covariance
        cov_beta = sd_beta ** 2 * Matern(length_scale=l_beta, nu=nu_beta)(grid.reshape(-1, 1))
        beta0 = np.random.multivariate_normal(np.zeros(t), cov_beta, p)
        beta1 = np.random.multivariate_normal(np.zeros(t), cov_beta, p)
        Y0 = X @ beta0
        Y1 = X @ beta1 + D
        Y = A[:,None] * Y1 + (1 - A[:,None]) * Y0 + np.random.normal(0, 1, (n,t))

        # True ATE
        true_ate = np.mean(Y1 - Y0, axis=0)

        #
        # OR estimator
        #

        # Fit linear regression models
        reg0 = LinearRegression().fit(X[A == 0], Y[A == 0])
        reg1 = LinearRegression().fit(X[A == 1], Y[A == 1])
        mu0 = reg0.predict(X)
        mu1 = reg1.predict(X)

        or_est = np.mean(mu1 - mu0, axis=0)

        # Fit FunGCN
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

        # Preprocess data
        X_repeated = np.repeat(X[:, :, np.newaxis], t, axis=2).transpose(1, 0, 2)
        B = np.concat([Y.reshape(1, n, t), X_repeated], axis=0)
        modalities = ['f'] + p * ['s']

        fun_gcn0 = FunGCN(data=B[:, A==0, :], y_ind=[0], var_modality=modalities, verbose=1)
        fun_gcn1 = FunGCN(data=B[:, A==1, :], y_ind=[0], var_modality=modalities, verbose=1)
        # preprocess the data
        fun_gcn0.preprocess_data(k_graph=k_graph, k_gcn=k_gcn, forecast_ratio=forecast_ratio,
                                test_size=test_size, val_size=val_size, random_state=seed)
        fun_gcn1.preprocess_data(k_graph=k_graph, k_gcn=k_gcn, forecast_ratio=forecast_ratio,
                                test_size=test_size, val_size=val_size, random_state=seed)
        # create graph
        fun_gcn0.graph_estimation(max_selected=max_selected, graph_path=save_graph_name)
        fun_gcn1.graph_estimation(max_selected=max_selected, graph_path=save_graph_name)
        # initialize GCN model
        fun_gcn0.initialize_gcn_model(pruning=pruning, nhid=nhid, dropout=dropout, kernel_size=kernel_size)
        fun_gcn1.initialize_gcn_model(pruning=pruning, nhid=nhid, dropout=dropout, kernel_size=kernel_size)
        # train GCN model
        fun_gcn0.train_model(lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, min_delta=min_delta)
        fun_gcn1.train_model(lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, min_delta=min_delta)
        # predict on test set
        prediction0 = fun_gcn0.predict(new_data=B).y_hat_os.reshape(n, t)
        prediction1 = fun_gcn1.predict(new_data=B).y_hat_os.reshape(n, t)

        or_est_gcn = np.mean(prediction1 - prediction0, axis=0)

        #
        # IPW estimator
        #

        # Fit logistic regression for propensity score
        ps_model = LogisticRegression().fit(X, A)
        ps_hat = ps_model.predict_proba(X)[:, 1]
        ipw_est = np.mean(A[:,None] * Y / ps_hat[:,None] - (1 - A[:,None]) * Y / (1 - ps_hat[:,None]), axis=0)

        #
        # DR estimator
        #

        dr_est = np.mean(A[:,None] * (Y - mu1) / ps_hat[:,None] + mu1 - (1 - A[:,None]) * (Y - mu0) / (1 - ps_hat[:,None]) - mu0, axis=0)
        dr_est_gcn = np.mean(A[:,None] * (Y - prediction1) / ps_hat[:,None] + prediction1 - (1 - A[:,None]) * (Y - prediction0) / (1 - ps_hat[:,None]) - prediction0, axis=0)

        # Save errors
        or_errors.append(np.mean((or_est - true_ate)**2))
        ipw_errors.append(np.mean((ipw_est - true_ate)**2))
        dr_errors.append(np.mean((dr_est - true_ate)**2))
        or_errors_gcn.append(np.mean((or_est_gcn - true_ate)**2))
        dr_errors_gcn.append(np.mean((dr_est_gcn - true_ate)**2))

        #
        # Misspecified features: nonlinear and reduced
        #

        X_miss = np.stack([
            np.sin(X[:, 0]),
            (X[:, 1] + X[:,2]) ** 2,
            np.log1p(np.abs(X[:, 3]))
        ], axis=1)

        # OR estimator (misspecified) - linear regression
        reg0_m = LinearRegression().fit(X_miss[A == 0], Y[A == 0])
        reg1_m = LinearRegression().fit(X_miss[A == 1], Y[A == 1])
        mu0_m = reg0_m.predict(X_miss)
        mu1_m = reg1_m.predict(X_miss)
        or_est_m = np.mean(mu1_m - mu0_m, axis=0)

        # OR estimator (misspecified) - FunGCN
        # Preprocess data
        X_repeated = np.repeat(X_miss[:, :, np.newaxis], t, axis=2).transpose(1, 0, 2)
        B = np.concat([Y.reshape(1, n, t), X_repeated], axis=0)
        modalities = ['f'] + X_miss.shape[1] * ['s']

        fun_gcn0 = FunGCN(data=B[:, A==0, :], y_ind=[0], var_modality=modalities, verbose=1)
        fun_gcn1 = FunGCN(data=B[:, A==1, :], y_ind=[0], var_modality=modalities, verbose=1)
        # preprocess the data
        fun_gcn0.preprocess_data(k_graph=k_graph, k_gcn=k_gcn, forecast_ratio=forecast_ratio,
                                test_size=test_size, val_size=val_size, random_state=seed)
        fun_gcn1.preprocess_data(k_graph=k_graph, k_gcn=k_gcn, forecast_ratio=forecast_ratio,
                                test_size=test_size, val_size=val_size, random_state=seed)
        # create graph
        fun_gcn0.graph_estimation(max_selected=max_selected, graph_path=save_graph_name)
        fun_gcn1.graph_estimation(max_selected=max_selected, graph_path=save_graph_name)
        # initialize GCN model
        fun_gcn0.initialize_gcn_model(pruning=pruning, nhid=nhid, dropout=dropout, kernel_size=kernel_size)
        fun_gcn1.initialize_gcn_model(pruning=pruning, nhid=nhid, dropout=dropout, kernel_size=kernel_size)
        # train GCN model
        fun_gcn0.train_model(lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, min_delta=min_delta)
        fun_gcn1.train_model(lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, min_delta=min_delta)
        # predict on test set
        prediction0_m = fun_gcn0.predict(new_data=B).y_hat_os.reshape(n, t)
        prediction1_m = fun_gcn1.predict(new_data=B).y_hat_os.reshape(n, t)

        or_est_m_gcn = np.mean(prediction1_m - prediction0_m, axis=0)

        # IPW estimator (misspecified)
        ps_model_m = LogisticRegression().fit(X_miss, A)
        ps_hat_m = ps_model_m.predict_proba(X_miss)[:, 1]
        ipw_est_m = np.mean(A[:,None] * Y / ps_hat_m[:,None] - (1 - A[:,None]) * Y / (1 - ps_hat_m[:,None]), axis=0)

        # DR estimator (misspecified)
        dr_est_mipw = np.mean(A[:,None] * (Y - mu1) / ps_hat_m[:,None] + mu1 - (1 - A[:,None]) * (Y - mu0) / (1 - ps_hat_m[:,None]) - mu0, axis=0)
        dr_est_mor = np.mean(A[:,None] * (Y - mu1_m) / ps_hat[:,None] + mu1_m - (1 - A[:,None]) * (Y - mu0_m) / (1 - ps_hat[:,None]) - mu0_m, axis=0)
        dr_est_mipw_gcn = np.mean(A[:,None] * (Y - prediction1) / ps_hat_m[:,None] + prediction1 - (1 - A[:,None]) * (Y - prediction0) / (1 - ps_hat_m[:,None]) - prediction0, axis=0)
        dr_est_mor_gcn = np.mean(A[:,None] * (Y - prediction1_m) / ps_hat[:,None] + prediction1_m - (1 - A[:,None]) * (Y - prediction0_m) / (1 - ps_hat[:,None]) - prediction0_m, axis=0)
        dr_est_mboth = np.mean(A[:,None] * (Y - mu1_m) / ps_hat_m[:,None] + mu1_m - (1 - A[:,None]) * (Y - mu0_m) / (1 - ps_hat_m[:,None]) - mu0_m, axis=0)
        dr_est_mboth_gcn = np.mean(A[:,None] * (Y - prediction1_m) / ps_hat_m[:,None] + prediction1_m - (1 - A[:,None]) * (Y - prediction0_m) / (1 - ps_hat_m[:,None]) - prediction0_m, axis=0)

        # Print results
        or_errors_m.append(np.mean((or_est_m - true_ate)**2))
        ipw_errors_m.append(np.mean((ipw_est_m - true_ate)**2))
        dr_errors_mipw.append(np.mean((dr_est_mipw - true_ate)**2))
        dr_errors_mor.append(np.mean((dr_est_mor - true_ate)**2))
        or_errors_m_gcn.append(np.mean((or_est_m_gcn - true_ate)**2))
        dr_errors_mipw_gcn.append(np.mean((dr_est_mipw_gcn - true_ate)**2))
        dr_errors_mor_gcn.append(np.mean((dr_est_mor_gcn - true_ate)**2))
        dr_errors_mboth.append(np.mean((dr_est_mboth - true_ate)**2))
        dr_errors_mboth_gcn.append(np.mean((dr_est_mboth_gcn - true_ate)**2))

    # make lists a melted dataframe with two columns: value and list name
    or_errors = np.array(or_errors)
    ipw_errors = np.array(ipw_errors)
    dr_errors = np.array(dr_errors)
    or_errors_gcn = np.array(or_errors_gcn)
    dr_errors_gcn = np.array(dr_errors_gcn)

    or_errors_m = np.array(or_errors_m)
    ipw_errors_m = np.array(ipw_errors_m)
    dr_errors_mipw = np.array(dr_errors_mipw)
    dr_errors_mor = np.array(dr_errors)
    dr_errors_mboth = np.array(dr_errors_mboth)
    or_errors_m_gcn = np.array(or_errors_m_gcn)
    dr_errors_mipw_gcn = np.array(dr_errors_mipw_gcn)
    dr_errors_mor_gcn = np.array(dr_errors_mor_gcn)
    dr_errors_mboth_gcn = np.array(dr_errors_mboth_gcn)

    data = {
        'OR': or_errors,
        'IPW': ipw_errors,
        'DR': dr_errors,
        'OR_m': or_errors_m,
        'IPW_m': ipw_errors_m,
        'DR_mipw': dr_errors_mipw,
        'DR_mor': dr_errors_mor,
        'DR_mboth': dr_errors_mboth,
        'OR_gcn': or_errors_gcn,  
        'DR_gcn': dr_errors_gcn,
        'OR_m_gcn': or_errors_m_gcn, 
        'DR_mipw_gcn': dr_errors_mipw_gcn,
        'DR_mor_gcn': dr_errors_mor_gcn,
        'DR_mboth_gcn': dr_errors_mboth_gcn
    }
    data = pd.DataFrame(data)

    data.to_csv('sim_results_DRFoS_explicit.csv', index=False)