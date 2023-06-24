import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

from src.helpers.lstm_ae import LSTMAutoencoder
from src.helpers.forecast import NadarayaWatson
from .functions import estimate_AR_res, winsor
from sklearn.linear_model import LinearRegression
from dcor import distance_correlation
from src.helpers.autoencoder import Autoencoder
from scipy.stats import chi2
from src.helpers.functions import lag_matrix
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.nonparametric.kernel_regression import KernelReg
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from sklearn.svm import SVR

def ar_forecast(p_AR_star_n, y_t, h, resids=False):
    """ Forecast AR model for h step ahead"""

    if p_AR_star_n > 0:
        a_hat, h_res = estimate_AR_res(y_t, h, p_AR_star_n)
        forecast = a_hat[0] + np.dot(a_hat[1:], y_t[-p_AR_star_n:])
    else:
        forecast = np.mean(y_t)

    if resids:
        return h_res
    else:
        return forecast

""" 
def ar_predict(y, p_AR_star_n, h=1):
    Tt = len(y)

    # Estimation
    Xt = np.ones((Tt-h, 1 + p_AR_star_n))
    #Xt[:, 1] = y[:-h]

    for i in range(p_AR_star_n):
        Xt[:, i+1] = y[i:Tt-h+i]
    


    Xt = Xt[:-h]
    Y = y[h:]

    b = np.linalg.lstsq(Xt, Y, rcond=None)[0]

    # Prediction
    YPred = np.zeros((1, 1))
    YPred[0, 0] = np.concatenate(([1], y[-1:])).dot(b)

    return YPred
    

def ar_predict(y, p_AR_star_n, h=1):
    Tt = len(y)
    # Estimation
    Xt = np.ones((Tt-p_AR_star_n, 1 + p_AR_star_n))
    
    for i in range(p_AR_star_n):
        Xt[:, i+1] = y[p_AR_star_n-1-i:Tt-1-i]

    Y = y[p_AR_star_n:]

    b = np.linalg.lstsq(Xt[h:], Y[:-h], rcond=None)[0]

    # Prediction
    YPred = np.zeros((1, 1))
    YPred[0, 0] = np.concatenate(([1], y[-p_AR_star_n:])).dot(b)

    return YPred
"""

def ar_predict(y, p, h):
    # Fit the model to the time series data
    model = AutoReg(y, lags=p)
    model_fit = model.fit()

    # Use the model to make a forecast of y_t+h
    forecast = model_fit.predict(start=len(y), end=len(y)+h-1)

    if h > 1:
        # Only return final forecast
        forecast = forecast[-1]

    return forecast

def scale_X(X_t, y_t, h, method = "regression"):
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
    else:
        raise ValueError("Scaling method not recognized")
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
    elif method == "sigmoid":
        n_components = hyper_params['n_components']
        gamma = hyper_params['gamma']
        pc = KernelPCA(n_components=n_components, kernel='sigmoid', gamma=gamma)
        X = pc.fit_transform(X)/gamma
    elif method == "rbf":
        n_components = hyper_params['n_components']
        gamma = hyper_params['gamma']
        pc = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma)
        X = pc.fit_transform(X)/gamma
    elif method == "ae":
        # Get the autoencoder model from input
        ae = dim_red_model

        # Fit the autoencoder to the data for a smaller number of epochs
        ae.train_model(X, num_epochs = hyper_params.get("update_epochs", 10), lr=hyper_params.get("update_lr", 0.001))

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

def loocv_ts(X, y, h = 1, p_AR_star_n = 1, method = "pca", scale_method = "distance_correlation", forecast_method = "ols", grid = None, ae=None):
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
    window = int(0.8 * T_train)
    N_test = T_train - window

    hyperparameters = grid.keys()
    parameter_values = grid.values()
    
    parameter_combinations = list(itertools.product(*parameter_values))

    if len(parameter_combinations) == 1:
        parameter_combinations = parameter_combinations[0]
        return dict(zip(hyperparameters, parameter_combinations))

    # Initialize the forecasting model
    lr = LinearRegression()

    # Initialize the arrays to store the forecasts
    y_hat = np.full((N_test - h, len(parameter_combinations)), np.nan)
    y_actual = np.full((N_test - h, 1), np.nan)

    rf = RandomForestRegressor()
    nw = NadarayaWatson()

    print("Number of model configurations: ", len(parameter_combinations))
    # Iterate over all model configurations
    for idx, hyper_params in enumerate(parameter_combinations):
        hyper_params = dict(zip(hyperparameters, hyper_params))


        # Print the current model configuration
        if method == "ae":
            # Train model on initial training window
            ae = Autoencoder(input_dim=X.shape[1], hyper_params=hyper_params)
            ae.train_model(X[:window], num_epochs = hyper_params.get("epochs", 200), lr=hyper_params.get("lr", 0.001))

        # Iterate over all windows
        for i in range(N_test - h):
            # Get the window of data
            X_t = X[:window + i]
            y_t = y[:window + i]
            y_actual[i, 0] = y[window + i + h - 1]

            # Scale the data
            scaling_factors = scale_X(X_t, y_t, h, scale_method)
            scaling_factors = winsor(np.abs(scaling_factors), p=(0, 90))

            X_t = X_t * scaling_factors

            # Reduce the dimensions
            X_t = reduce_dimensions(X_t, method, hyper_params, dim_red_model=ae)

            if p_AR_star_n > 0:
                # Add lags of y to x
                X_t = lag_matrix(X_t, y_t, p_AR_star_n)
                # Remove the first p_AR_star_n observations of y_t
                y_t = y_t[p_AR_star_n-1:]

            # Forecast 
            if forecast_method == "ols":
                lr.fit(X_t[:-h], y_t[h:])
                y_hat[i, idx] = lr.predict(X_t[-1].reshape(1, -1))
            elif forecast_method == "rf":
                rf.fit(X_t[:-h], y_t[h:])
                y_hat[i, idx] = rf.predict(X_t[-1].reshape(1, -1))
            elif forecast_method == "nw":
                nw.fit(X_t[:-h], y_t[h:])
                y_hat[i, idx] = nw.predict(X_t[-1].reshape(1, -1))
            elif forecast_method == "gam":
                gam = LinearGAM().fit(X_t[:-h], y_t[h:])
                y_hat[i, idx] = gam.predict(X_t[-1].reshape(1, -1))
            elif forecast_method == "svr":
                svr = SVR(kernel="rbf").fit(X_t[:-h], y_t[h:])
                y_hat[i, idx] = svr.predict(X_t[-1].reshape(1, -1))
            elif forecast_method == "krr":
                krr = KernelRidge(kernel="rbf").fit(X_t[:-h], y_t[h:])
                y_hat[i, idx] = krr.predict(X_t[-1].reshape(1, -1))
            else:
                raise ValueError("Invalid forecast method")
        

        
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

def loocv_ts_bayes(X, y, forecast_method, h = 1, p_AR_star_n = 1, method = "pca", scale_method = "distance_correlation", space = None, ae=None, trials= 50):
    """ 
    Leave one out cross validation for time series with hyperparameter optimization

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
    space: dictionary
        Space of values to search over for hyperopt
    """
    space_hp = {
    'hidden_dim': hp.choice('hidden_dim', space['hidden_dim']),
    'layer_dims': hp.choice('layer_dims', space['layer_dims']),
    'gauss_noise': hp.uniform('gauss_noise', 0.01, 0.75),
    'dropout': hp.uniform('dropout', 0.01, 0.5),
    'epochs': hp.choice('epochs', space['epochs']),
    'batch_size': hp.choice('batch_size', space['batch_size']),
    'update_epochs': hp.choice('update_epochs', space['update_epochs']),
    'update_lr': hp.choice('update_lr', space['update_lr']),
    'lr': hp.choice('lr', space['lr']),	
    }

    T_train = X.shape[0]
    window = int(0.8 * T_train)
    N_test = T_train - window
    
    # Initialize the forecasting model
    lr = LinearRegression()
    svr = SVR(kernel="rbf")
    rf = RandomForestRegressor(n_estimators=100)

    # Initialize the arrays to store the forecasts
    y_hat = np.full((N_test - h, 1), np.nan)
    y_actual = np.full((N_test - h, 1), np.nan)
    
    def objective(hyper_params):
        # Print the current model configuration
        if method == "ae":
            # Train model on initial training window
            ae = Autoencoder(input_dim=X.shape[1], hyper_params=hyper_params)
            ae.train_model(X[:window], num_epochs = hyper_params.get("epochs", 100), lr=hyper_params.get("lr", 0.001))

        # Iterate over all windows
        for i in range(N_test - h):
            # Get the window of data
            X_t = X[:window + i]
            y_t = y[:window + i]
            y_actual[i, 0] = y[window + i + h - 1]

            # Scale the data
            scaling_factors = scale_X(X_t, y_t, h, scale_method)
            scaling_factors = winsor(np.abs(scaling_factors), p=(0, 90))

            X_t = X_t * scaling_factors

            # Reduce the dimensions
            X_t = reduce_dimensions(X_t, method, hyper_params, dim_red_model=ae)

            if p_AR_star_n > 0:
                # Add lags of y to x
                X_t = lag_matrix(X_t, y_t, p_AR_star_n)
                # Remove the first p_AR_star_n observations of y_t
                y_t = y_t[p_AR_star_n-1:]

            # Forecast 
            if forecast_method == "ols":
                lr.fit(X_t[:-h], y_t[h:])
                y_hat[i, 0] = lr.predict(X_t[-1].reshape(1, -1))
            elif forecast_method == "svr":
                svr.fit(X_t[:-h], y_t[h:])
                y_hat[i, 0] = svr.predict(X_t[-1].reshape(1, -1))
            elif forecast_method == "rf":
                rf.fit(X_t[:-h], y_t[h:])
                y_hat[i, 0] = rf.predict(X_t[-1].reshape(1, -1))
            else:
                raise ValueError("Invalid forecast method")

        
        # Compute MSE for each model configuration
        result = ((y_actual - y_hat)**2).mean()

        return result

    # Use Bayesian optimization to find the best hyperparameters
    best = fmin(fn=objective, space=space_hp, algo=tpe.suggest, max_evals=trials)
    print(best)
    best = {k: space[k][v] if k in space else v for k, v in best.items()}

    # Print the best model configuration
    print("Best model configuration: ", best)
        
    return best 



