import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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

def sPCAest(target, X, nfac, quantile=[0, 90], h_steps=1):
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
