import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.ensemble import RandomForestRegressor
import fasttreeshap
from .functions import estimate_AR_res
from sklearn.linear_model import LinearRegression
from dcor import distance_correlation
from minepy import cstats

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

    # Get residuals of regressing y_t on lagged y_t
    res_h = ar_forecast(p_AR_star_n, y_t, h, resids=True)

    if method == "regression":
        lr = LinearRegression()

        # Compute the betas for scaling the variables
        for j in range(X_t.shape[1]):
            lr.fit(X_t[:-h, j].reshape(-1, 1), y_t[h:])
            scaling_factors[j] = lr.coef_[0]
    elif method == "distance_correlation":
        for j in range(X_t.shape[1]):
            scaling_factors[j] = distance_correlation(X_t[:-h, j], y_t[h:])
    elif method == "mic":
        raise NotImplementedError
        for j in range(X_t.shape[1]):
            scaling_factors[j], _ = cstats(X_t[:-h, j].reshape(1, -1), y_t[h:].reshape(1, -1))
    elif method == "shap":
        raise NotImplementedError
        if model:
            rf = model
        else:
            rf = RandomForestRegressor(verbose=1)
            rf.fit(X_t[:-h], y_t[h:])
        explain = fasttreeshap.TreeExplainer(rf)
        scaling_factors = explain.shap_values(X_t[:-h], y_t[h:]).mean(axis=0)

    # Return scaling factors
    return scaling_factors

def reduce_dims(X, method, nfac, kernel_params=None):
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
        pc = PCA(n_components=nfac)
        X = pc.fit_transform(X)
    elif method == "kpca":
        pc = KernelPCA(n_components=nfac, kernel="poly", degree=5)
        X = pc.fit_transform(X)
    elif method == "ae":
        # TODO: Implement autoencoder
        raise NotImplementedError
    
    return X    

def loocv_ts(X, y, h = 1, p_AR_star_n = 1, method = "pca", scale_method = "distance_correlation"):
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
    grid: array
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

    grid = {"nfac": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "kernel": ["poly", "rbf", "sigmoid"],
            "degree": [2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
    results = {}
    lr = LinearRegression()

    # Iterate over all model configurations
    for nfac in grid["nfac"]:
        y_hat = np.full(N_test, np.nan)
        y_actual = np.full(N_test, np.nan)

        # Iterate over all windows
        for i in range(N_test):
            # Get the window of data
            X_t = X[i:window + i]
            y_t = y[i:window + i]

            # Scale the data
            scaling_factors = scale_X(X_t, y_t, h, p_AR_star_n, scale_method)
            X_t = X_t * scaling_factors

            # Reduce the dimensions
            X_t = reduce_dims(X_t, method, nfac)

            # Forecast
            lr.fit(X_t[:-h], y_t[h:])
            y_hat = lr.predict(X_t[-h:])

            # Store the actual value
            y_actual[i] = y_t[-1]
        
        # Store and compute MSE
        results[nfac] = np.mean((y_actual - y_hat)**2)
        print(f"nfac: {nfac}, MSE: {results[nfac]}")

    # Return the model configuration with the lowest MSE
    best_nfac = min(results, key=results.get)

    return best_nfac
    








        
