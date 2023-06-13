import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
import fasttreeshap

from src.helpers.lstm_ae import LSTMAutoencoder
from .functions import estimate_AR_res
from sklearn.linear_model import LinearRegression
from dcor import distance_correlation
from minepy import cstats
from src.helpers.autoencoder import Autoencoder
from scipy.stats import chi2


def ar_forecast(p_AR_star_n, y_t, h, resids=False):
    """ Forecast AR model for h step ahead"""

    if p_AR_star_n >= 0:
        a_hat, h_res = estimate_AR_res(y_t, h, 1)
        forecast = a_hat[0] + np.dot(a_hat[1:], y_t[-1:])
    else:
        forecast = np.mean(y_t)

    if resids:
        return h_res
    else:
        return forecast
    

def scale_X(X_t, y_t, h,p_AR_star_n, method = "regression", model=None):
    scaling_factors = np.full(X_t.shape[1], np.nan)
    X_t = X_t.copy()
    y_t = y_t.copy()

    if method == "regression":
        lr = LinearRegression()

        # Compute the betas for scaling the variables
        for j in range(X_t.shape[1]):
            lr.fit(X_t[:-h, j].reshape(-1, 1), y_t[h:])
            scaling_factors[j] = lr.coef_[0]
    elif method == "distance_correlation":
        for j in range(X_t.shape[1]):
            scaling_factors[j] = distance_correlation(X_t[:-h, j], y_t[h:])
    elif method == "none":
        scaling_factors = np.ones(X_t.shape[1])
    
    # Return scaling factors
    return scaling_factors

def reduce_dimensions(X, method, hyper_params, dim_red_model=None, cv=False):
    """ 
    Applies dimension reduction to X with method and nfac

    Parameters
    ----------
    X : array
        Array of features
    method : string
        Method to use for dimension reduction
    nfac : int
        Number of factors to reduce to

    Returns
    -------
    X : array
        Array of features with reduced dimensions
    """

    if method == "pca":
        nfac = hyper_params['nfac']
        pc = PCA(n_components=nfac)
        X = pc.fit_transform(X)
    elif method == "kpca":
        n_components = hyper_params['n_components']
        kernel = hyper_params['kernel']
        gamma = hyper_params['gamma']
        pc = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
        X = pc.fit_transform(X)/gamma
    elif method == "ae":
        # Get the autoencoder model from input
        ae = dim_red_model

        # Fit the autoencoder to the data for a smaller number of epochs
        ae.train_model(X, num_epochs = hyper_params.get("update_epochs", 50), lr=hyper_params.get("update_lr", 0.001))

        X = ae.encode(X)
    elif method == "lstm":
        if cv:
            # Initialize the autoencoder model with the hyperparameters
            lstm = LSTMAutoencoder(input_dim=X.shape[1], hyper_params=hyper_params)

            # Fit the autoencoder to the data for a large number of epochs
            lstm.train_model(X, num_epochs = hyper_params.get('epochs', 200), lr=hyper_params.get("lr", 0.001))
        else:
            # Get the autoencoder model from input
            lstm = dim_red_model

            # Fit the autoencoder to the data for a smaller number of epochs
            lstm.train_model(X, num_epochs = 20, lr=hyper_params.get("lr", 0.001))

        # Reduce the dimensions of the data
        X = lstm.encode_offline(X) 
    elif method == "none":
        pass

    return X   

def loocv_ts(X, y, h = 1, p_AR_star_n = 1, method = "pca", scale_method = "distance_correlation", grid = None, ae=None):
    """ 
    Leave one out cross validation for time series

    Parameters
    ----------
    X : array
        Array of features
    y : array
        Array of target values
    h : int
        Forecast horizon
    p_AR_star_n : int
        Order of AR model
    method : string
        Method to use for dimension reduction
    grid: dictionary
        Grid of values to search over
    """
    """ 
    For a given method, start at some window and create 1 forecast
    Then move the window forward and create another forecast and so on
    until the end of the data is reached

    Store the MSE for the model 
    
    Repeat for all model configurations

    Return the model configuration with the lowest MSE
    """
    T_train = X.shape[0]
    window = int(0.6 * T_train)
    N_test = T_train - window

    hyperparameters = grid.keys()
    parameter_values = grid.values()
    
    parameter_combinations = list(itertools.product(*parameter_values))
        
    # Initialize the forecasting model
    lr = LinearRegression()

    # Initialize the arrays to store the forecasts
    y_hat = np.full((N_test, len(parameter_combinations)), np.nan)
    y_actual = np.full((N_test, 1), np.nan)
    
    print("Number of model configurations: ", len(parameter_combinations))
    
    # Iterate over all model configurations
    for idx, hyper_params in enumerate(parameter_combinations):
        hyper_params = dict(zip(hyperparameters, hyper_params))

        # Print the current model configuration
        if method == "ae":
            print("Model configuration: ", hyper_params, " ", idx + 1, "/", len(parameter_combinations))

            # Train model on initial training window
            ae = Autoencoder(input_dim=X.shape[1], hyper_params=hyper_params)
            ae.train_model(X[:window], num_epochs = hyper_params.get("epochs", 100), lr=hyper_params.get("lr", 0.001))

        # Iterate over all windows
        for i in range(N_test):
            # Get the window of data
            X_t = X[:window + i]
            y_t = y[:window + i]
            y_actual[i, 0] = y[window + i]

            # Scale the data
            scaling_factors = scale_X(X_t, y_t, h, p_AR_star_n, scale_method)
            X_t = X_t * scaling_factors

            # Reduce the dimensions
            X_t = reduce_dimensions(X_t, method, hyper_params, dim_red_model=ae)

            # Forecast 
            lr.fit(X_t[:-h], y_t[h:])
            y_hat[i, idx] = lr.predict(X_t[-1].reshape(1, -1))
        
    # Compute MSE for each model configuration
    results = ((y_actual - y_hat)**2).mean(axis=0)
    
    # Return the best model configuration
    best_idx = np.argmin(results)
    best_params = parameter_combinations[best_idx]
    best_params = dict(zip(hyperparameters, best_params))

    # Print the best model configuration
    print("Best model configuration: ", best_params)

    return best_params    

def standardize(X):
    """ Standardize the data """
    return (X - X.mean(axis=0)) / X.std(axis=0)

def forecast(x, y, h, method="ols", hyper_params=None):
    """
    Given a set of features and target values, compute the forecast using the given method. NB: x, y will be shifted by h in the function
    """
    if method == "ols":
        model = LinearRegression()
    elif method == "krr":
        model = KernelRidge(kernel_params=hyper_params)
    elif method == "rf":
        model = RandomForestRegressor(hyper_params)
    else:
        raise ValueError("Unknown method")

    # Estimate regression coefficients
    model.fit(x[:-h], y[h:])
    
    # Compute the forecast of the PCA and scaled PCA model
    prediction = model.predict(x[-1].reshape(1, -1))

    return prediction


def get_krr_grid(y, X):
    """ 
    Get the grid of hyperparameters for the kernel ridge regression model according to Exterkate

    """
    N = X.shape[1]

    c_N = chi2.ppf(0.95, N)
    sigma_0 = np.sqrt(c_N ) / np.pi

    # Reduce dimension of data using PCA using first 4 PCs
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)

    # Get the R2 of regressing y on the first 4 PCs
    r2 = LinearRegression().fit(X_pca, y).score(X_pca, y)

    lambda_0 = (1 - r2) / r2

    # Get the grid of hyperparameters
    grid = {"lambda": [lambda_0/8, lambda_0/4, lambda_0/2, lambda_0, 2*lambda_0], "sigma": [sigma_0/2, sigma_0, 2*sigma_0, 4*sigma_0, 8*sigma_0]}

    return grid


