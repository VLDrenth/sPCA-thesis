import numpy as np
from scipy.stats import chi2

def linear_reg(y, x, constant=1, method='NW', nlag=0):
    """
    Estimaete a linear regression model using Ordinary Least Squares (OLS) with standard errors computed
    according to a user-specified method and, if applicable, with a user-specified number of lags in the computation.
    RETURNS:

    parm:       K x nVar array of regression coefficients
    std_err:    K x nVar array of standard errors
    t_stat:     K x nVar array of t-statistics
    reg_se:     nVar x 1 array of regression standard errors
    adj_r2:     nVar x 1 array of adjusted R^2 values
    bic:        nVar x 1 array of BIC values
    """


    # Make sure that y and x are numpy arrays
    x = np.array(x)
    y = np.array(y)

    y = y.reshape((-1, 1))

    # Error checking on input 
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError('linear_reg: The dimensions of y and x should be 2.',
                        'Given is {}'.format(y.shape))
   
    if x.shape[0] != y.shape[0]:
        raise ValueError('linear_reg: Length of y and x is not the same.', 
                         'Given is x: {} and y: {}'.format(x.shape[0], y.shape[0]))
    
    if method not in ['OLS', 'W', 'NW', 'HH', 'Skip']:
        raise ValueError('linear_reg: Wrong specification for standard errors provided.')
    
    # Error check on lag number or set to zero if inconsequential
    if (method in ['NW', 'HH']) and nlag is None:
        raise ValueError('linear_reg: Lag length unspecified.')
    
    # Setting default parameters
    if constant == 1:
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    
    # Preliminaries
    T, nVar = y.shape
    K = x.shape[1]
    Exx = np.dot(x.T, x) / T
    parm = np.linalg.lstsq(x, y, rcond=None)[0]
    errv = y - np.dot(x, parm)
    reg_se = np.sum(errv**2, axis=0) / T
    bic = np.log(reg_se) + K * np.log(T) / T
    
    # Compute standard errors according to method
    std_err = np.empty((K, nVar))
    t_stat = np.empty((K, nVar))
    
    # Newey-West standard errors
    errv_lag = np.concatenate((np.zeros((nlag, nVar)), errv))

    omega = np.zeros((K, K))
    for i in range(nlag):
        omega.fill(0)
        for t in range(nlag, T + nlag):
            xt = x[t - nlag, :].reshape((1, K))
            et = errv_lag[t - i, :].reshape((nVar, 1))
            omega += np.dot(np.dot(xt.T, et), np.dot(et.T, xt))
        omega /= T

        vcov = np.linalg.inv(Exx) @ omega @ np.linalg.inv(Exx)
        std_err = np.sqrt(np.diag(vcov))
        t_stat = parm / std_err

    vary = np.mean((y - np.ones((T, 1)) * np.mean(y))**2)
    adj_r2 = (1 - (reg_se / vary) * (T - 1) / (T - K))

    return parm, std_err, t_stat, reg_se, adj_r2, bic



            
            




