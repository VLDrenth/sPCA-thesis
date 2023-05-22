import numpy as np

def ARDL_multi(y, z, h, p):
    """ 
    Estimates intercept/slope parameters for the ARDL(p1,p2) model

    y(t,h) = a(0) + a(1)*y(t-1) + ... + a(p1)*y(t-p1) + ...
            b(1)*z(t-1) + ... + b(p2)*z(t-p2) + e(t,h),

    where

    y(t,h) = horizon-h mean of y = (1/h)*sum_(j=1)^(h)y(t+(j-1)) 

    Input

    y = vector of y observations
    z = vector of z observations
    h = forecast horizon
    p = vector of ARDL lags (p1,p2)

    Output
    
    c_hat = (p1+p2+1)-vector of coefficient estimates
            [a(0),a(1),...,a(p1),b(1),...,b(p2)]
    """
    # Take care of Preliminaries
    sz = z.shape[1]
    T = y.shape[0]
    y_h = np.zeros(T - (h - 1))
    for t in range(T - (h - 1)):
        y_h[t] = np.mean(y[t : t + (h - 1)])

    # Create regressand/regressors
    p1 = p[0]
    p2 = p[1]
    p_max = max(p1, p2)
    y_h = y_h[p_max:]
    y_lags = np.empty((len(y_h), p_max))
    z_lags = np.empty((len(y_h), p_max * sz))
    for j in range(1, p_max + 1):
        y_lags[:, j - 1] = y[p_max - j : T - j - (h - 1)]
        z_lags[:, (j - 1) * sz : j * sz] = z[p_max - j : T - j - (h - 1), :]
    
    if p1 == 0:
        Z = np.concatenate((np.ones((len(y_h), 1)), z_lags), axis=1)
    else:
        Z = np.concatenate((np.ones((len(y_h), 1)), y_lags[:, :p1], z_lags[:, :p2*sz]), axis=1)

    # Estimating parameters
    c_hat = np.linalg.inv(Z.T @ Z) @ (Z.T @ y_h)

    return c_hat
