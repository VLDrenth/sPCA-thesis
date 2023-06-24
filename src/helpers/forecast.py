import itertools
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
from sklearn.svm import SVR

class NadarayaWatson:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        return self
    
    def predict(self, X):
        if self.kernel == 'gaussian':
            K = np.exp(-0.5 * cdist(X, self.X)**2 / self.bandwidth**2)
        else:
            raise ValueError('Kernel function not recognized.')
            
        return np.sum(K * self.y, axis=1) / np.sum(K, axis=1)
    
    def set_params(self, **params):
        if 'kernel' in params:
            self.kernel = params['kernel']
        if 'bandwidth' in params:
            self.bandwidth = params['bandwidth']
        return self


class Forecast:

    def __init__(self, method, hyper_params=None, h=1):
        self.method = method
        if not h:
            print("Warning: h is not set, defaulting to 1")

        if method == "ols":
            self.model = LinearRegression()
        elif method == "krr":
            self.model = KernelRidge()
        elif method == "rf":
            self.model = RandomForestRegressor(n_estimators=100)
        elif method == "nw":
            self.model = NadarayaWatson()
        elif method == "svr":
            self.model = SVR(kernel="rbf")
        else:
            raise ValueError("Unknown method")
        
        self.hyper_params = hyper_params
        self.h = h

    def predict(self, x, y):
        """ Predicts the next value from factors"""
        if self.method == "poly":
            x = PolynomialFeatures(degree=2).fit_transform(x)

        # Estimate regression coefficients
        self.model.fit(x[:-self.h], y[self.h:])

        # Compute the forecast of the PCA and scaled PCA model
        prediction = self.model.predict(x[-1].reshape(1, -1))

        return prediction
    
    def cross_validate(self, X, y, hyper_params):
        """ Cross validate the model """
        T_train = X.shape[0]
        window = int(0.7 * T_train)
        N_test = T_train - window
        h = self.h

        hyperparameters = hyper_params.keys()
        parameter_values = hyper_params.values()
        
        parameter_combinations = list(itertools.product(*parameter_values))
        
        # Initialize the arrays to store the forecasts
        y_hat = np.full((N_test - h, len(parameter_combinations)), np.nan)
        y_actual = np.full((N_test - h, 1), np.nan)
        
        # Iterate over all model configurations
        for idx, hyper_params in enumerate(parameter_combinations):
            hyper_params = dict(zip(hyperparameters, hyper_params))

            # Print the current model configuration
            print("Model configuration: ", hyper_params, " ", idx + 1, "/", len(parameter_combinations))

            # Iterate over all windows
            for i in range(N_test - h):
                # Get the window of data
                X_t, y_t = X[i:window + i], y[i:window + i]
                y_actual[i, 0] = y[window + i + h -1]
                
                # Make the prediction using the current model configuration
                self.model.fit(X_t[:-h], y_t[h:])
                y_hat[i, idx] = self.model.predict(X_t[-1].reshape(1, -1))
            
        # Compute MSE for each model configuration
        results = ((y_actual - y_hat)**2).mean(axis=0)
        
        # Return the best model configuration
        best_idx = np.argmin(results)
        best_params = parameter_combinations[best_idx]
        best_params = dict(zip(hyperparameters, best_params))

        print("Best model configuration: ", best_params, " with MSE: ", results[best_idx])

        self.model = self.model.set_params(**best_params)

        return best_params    


        
        

