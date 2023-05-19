from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, FastICA
import numpy as np
import pandas as pd

def winsor(x, p):
    if not np.ndim(x) == 1:
        print("Got x of shape" + str(x.shape) + " dimensions, expected 1")
        raise ValueError('Input argument "x" must be a vector')
    if len(p) != 2:
        raise ValueError('Input argument "p" must be a 2*1 vector')
    if not all(0 <= i <= 100 for i in p):
        raise ValueError('Cut-off percentiles must be in [0,100] range')
    if p[0] > p[1]:
        raise ValueError('Left cut-off percentile exceeds right cut-off percentile')
    p = np.percentile(x, p)

    i1 = x < p[0]
    v1 = np.min(x[~i1])
    i2 = x > p[1]
    v2 = np.max(x[~i2])
    y = x.copy()
    y[i1] = v1
    y[i2] = v2
    
    if len(np.shape(y)) == 2:
        y = y[:, 0]
    if len(np.shape(x)) == 2:
        x = x[:, 0]
    if np.shape(y) != np.shape(x):
        raise ValueError('Error in dimensions of the input and output vectors')
    if np.any(np.isnan(x)):
        raise ValueError('Input vector contains NaN values')
    if np.any(np.isnan(y)):
        raise ValueError('Output vector contains NaN values')
    if np.any(np.isinf(x)):
        raise ValueError('Input vector contains infinite values')
    if np.any(np.isinf(y)):
        raise ValueError('Output vector contains infinite values')
    if np.any(y == np.inf):
        raise ValueError('Output vector contains infinite values')
    if np.any(y == -np.inf):
        raise ValueError('Output vector contains infinite values')
    
    out = y
    # if nargout > 1: ??
    return out

def sPCAest(target, X, nfac, quantile=[0, 100], h_steps=1):
    """
    This function performs sPCA (scaled principal component analysis) on the input data.
    It takes in three input variables, X, target, and nfac, and returns the sPCA factors in f.
    """

    T = len(target)
    if len(X) != T:
        print(f"X is of length {len(X)} and Y is of length {T}")
        raise ValueError('X and Y variables not of equal length')

    # Standardize X to Xs
    Xs = np.zeros(X.shape)
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    beta = np.empty(Xs.shape[1])
    for j in range(Xs.shape[1]):
        lr = LinearRegression(fit_intercept=True)

        # Drop the last h_steps observations from Xs and target to avoid look-ahead bias
        lr.fit(Xs[:-h_steps, j].reshape(-1, 1), target[:-h_steps])
        beta[j] = lr.coef_[0]

        #xvar = np.column_stack((np.ones(T), Xs[:, j]))
        #parm = np.linalg.lstsq(xvar, target, rcond=None)[0]
        #beta[0, j] = parm[1]
    
    if quantile[0] != 0 or quantile[1] != 100:
        beta = winsor(np.abs(beta.flatten()), quantile)

    # Scale Xs by the estimated beta
    scaleXs = np.empty((Xs.shape[0], Xs.shape[1]))
    for j in range(Xs.shape[1]):
        scaleXs[:, j] = Xs[:, j] * beta[j]

    # Perform PCA on the scaled Xs
    pca = PCA(n_components=nfac)
    pca.fit(scaleXs)
    f = pca.transform(scaleXs)
    eigen_values = pca.explained_variance_ratio_
    #ehat, f, lambda_mat, ve2, eigen_values = pc_T(scaleXs, nfac)
    return  f, eigen_values/np.sum(eigen_values)

def spca_is(X, target, nfac=5, scale=True, quantile=[0, 100]):
    """In sample prediction using sPCA"""

    if scale:
        # Obtain factors from sPCA
        factors, eigen_values = sPCAest(target, X, nfac=nfac, quantile=quantile)
    else:
        # Normalize X
        Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

        # Obtain factors from PCA
        pca = PCA(n_components=nfac)
        factors = pca.fit_transform(Xs)
        eigen_values = pca.explained_variance_ratio_

    # Fit the model on the factors obtained on the training set
    lr = LinearRegression(fit_intercept=True)

    lr.fit(factors, target)

    # Predict the response variables on the factors from the training set
    y_pred = lr.predict(factors)

    # Return predicted values, and eigen values
    return y_pred, eigen_values


def pc_T(y, nfac):
    bigt, bign = y.shape
    
    # Calculate the inner product of y with itself to get yy
    yy = np.dot(y, y.T)
    
    # Perform eigendecomposition on yy
    eigval, Fhat0 = np.linalg.eig(yy)
    
    # Sort the eigenvalues in descending order and sort the eigenvectors accordingly
    eigval_desc_order = np.argsort(eigval)[::-1]
    eigval_sorted = np.real(eigval[eigval_desc_order])

    Fhat0_sorted = np.real(Fhat0[:, eigval_desc_order])
    
    # Calculate the estimated factor scores fhat
    fhat = np.dot(Fhat0_sorted[:, :nfac], np.sqrt(bigt))
    
    # Calculate the estimated loading matrix lambda
    lambda_mat = np.dot(y.T, fhat) / bigt
    
    # Calculate the estimated error matrix ehat
    ehat = y - np.dot(fhat, lambda_mat.T)
    
    # Calculate the estimated residual variance for each variable in y
    ve2 = np.sum(ehat**2, axis=1) / bign
    
    # Store the sorted eigenvalues in ss and return all calculated values
    ss = eigval_sorted
    
    return ehat, fhat, lambda_mat, ve2, ss

def compute_R2(actuals, preds, adjusted = True, nfac = None):
    """Compute R2 and adjusted R2"""
     
    if adjusted and nfac is None :
        raise ValueError("Need to give number of factors")
    
    SSR_list = []
    SST_list = []
    mean_actuals = np.mean(actuals)

    for i in range(len(actuals)):
        SSR_list.append((actuals[i] - preds[i])**2)
        SST_list.append((actuals[i] - mean_actuals)**2)

    SSR = np.sum(SSR_list)
    SST = np.sum(SST_list)

    print("SSR: ", SSR, "SST: ", SST)
    R2 = 1 - SSR/SST

    if not adjusted:
        return R2
    else:
        n = len(actuals)
        k = nfac
        R2_adj = 1 - (1 - R2) * (n - 1) / (n - k - 1)
        
        return R2_adj
     

def R2_OS(actuals, forecasts_pca, forecasts_benchmark):
    """Compute R2 for out of sample predictions"""
    
    actuals = np.array(actuals)
    forecasts_pca = np.array(forecasts_pca)
    forecasts_benchmark = np.array(forecasts_benchmark)

    SSR_pca = np.sum((actuals - forecasts_pca)**2)
    SSR_benchmark = np.sum((actuals - forecasts_benchmark)**2)

    R2_os = 1 - SSR_pca/SSR_benchmark
    R2_os = np.round(R2_os * 100, 2)

    return R2_os


def AR_predict(series, max_lags=20):
    ar_preds = np.zeros(len(series))

    for i in range(1,max_lags):
        ar_preds[i] = np.mean(series[:i])

    for i in range(max_lags, len(series)):
        # Choose optimal lag length
        ar = fit_lags(series[:i], 10)
        
        model_fit = ar.fit()    
        ar_preds[i] = model_fit.predict(start=i, end=i)[0]

    return ar_preds

def corr_uniform(size=5, rho = 0.55):
    # Generate N samples from a multivariate normal distribution
    samples = np.random.multivariate_normal([0, 0], cov=[[1, rho], [rho, 1]], size=size)

    # Transform the samples into uniform distributions
    u1 = (np.sin(samples[:, 0]) + 1) / 2
    u2 = (np.sin(samples[:, 1]) + 1) / 2

    return u1, u2

def generate_data(n, T, N, h_steps=1, heteroskedastic=False, psi_max=None, rho=None):
    # Check if weak or strong factor model
    if n == N:
        # Strong factor model, require psi_max and rho
        assert psi_max is not None
        assert rho is not None

    # Generate y_{t + h} = g_t + e_{t + h}
    # e_{t + h} ~ N(0, 1)
    # g_t ~ N(0, 1)
    # X_t,i = g_t*phi_i + h_t*psi_i + u_t,i

    # Set T to T + h to account for lag
    T_plus_h = T + h_steps

    # Generate g_t
    g = np.random.normal(0, 1, T_plus_h)

    # Generate h_t
    h = np.random.normal(0, 1, T_plus_h)

    # Generate e_t+h
    e = np.random.normal(0, 1, T_plus_h)

    # Generate y_{t+h}
    y_h = g + e

    # Generate y_t
    y_t = np.zeros(T_plus_h)
    for i in range(h_steps, T_plus_h):
        y_t[i] = y_h[i - h_steps]

    
    phi = np.zeros(N)
    psi = np.zeros(N)

    if n < N:
        # Generate phi_i (Nx1) with n < N nonzero elements
        phi[:n] = np.random.uniform(0, 1, n)

        # Generate psi_i (Nx1) with n < N nonzero elements
        psi[:n] = np.random.uniform(0, 1, n)

        # Generate idiosyncratic error's variances
        variance_u = np.random.uniform(0, 1, N)        

    elif n == N:
        # Phi is U(0, phi_max)
        psi = np.random.uniform(0, psi_max, N)

        # Draw correlated phi and sigma_u
        # Generate correlation matrix for phi and sigma_u
        phi, variance_u = corr_uniform(rho=rho*1.1, size=N)

    u = np.zeros((T, N))
    # Generate u_t,i depending on whether the model is heteroskedastic or not
    if heteroskedastic:
        for t in range(T):
            scale = np.random.uniform(0.5, 1.5)
            u[t, :] = np.random.normal(0, variance_u * scale, N)
    else:
        # Generate covariance matrix for u_t
        sigma_u = np.diag(variance_u)

        # Generate u
        u = np.random.multivariate_normal(np.zeros(N), sigma_u, T)

    # Drop first h rows
    y_h = y_h[h_steps:]
    y_t = y_t[h_steps:]
    e = e[h_steps:]
    g = g[h_steps:]
    h = h[h_steps:]

    # Generate X_t
    X = np.zeros((T, N))
    for i in range(N):
        X[:, i] = g*phi[i] + h*psi[i] + u[:, i]

    results = {'y_t':y_t,
               'y_h':y_h,
                'X':X,
                'g':g,
                'e':e,
                }
    
    return results
