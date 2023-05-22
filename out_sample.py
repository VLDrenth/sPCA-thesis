import numpy as np
from sklearn.linear_model import LinearRegression
from src.helpers.functions import get_data, pc_T, predict_pca, estimate_AR_res


def out_sample(X, y):
    h = 1
    T = y.shape[0]
    nfac = 5

    M = (1984-1959)*12  # In sample periods
    N = T - M  # Out of sample periods

    error_pca = np.zeros((N - h, 1))  # Forecast errors of PCA 
    error_spca = np.zeros((N - h, 1))  # Forecast errors of scaled PCA
    error_ar = np.zeros((N - h, 1))  # Forecast errors of AR model
    p_AR_star_n = 1  # Number of lags for AR(p) model

    # Estimate the AR model
    a_hat, res_h = estimate_AR_res(y, h, p_AR_star_n)
    error_ar = res_h[-N:]

    # Estimate the PCA model
    for n in range(N - h):
        print(n)

        # Use all available data up to time t
        X_train = X[:M + n, :]
        y_train = y[:M + n]

        y_h = np.mean(y[M + n + 1:M + n + 1 + h])

        # Standardize the data
        X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

        # Compute the principal components
        _, x_pc, _, _, _ = pc_T(X_train, nfac)

        # Estimate regression coefficients
        reg = LinearRegression()
        reg.fit(x_pc[:-h,:], y_train[:-h])

        # Compute the forecast
        y_hat = reg.predict(x_pc[-1,:].reshape(1, -1))

        # Compute the forecast error
        error_pca[n] = y_h - y_hat

    # Compute the R squared out of sample against the AR model
    SSE_pca = np.sum(error_pca**2)
    SSE_ar = np.sum(error_ar**2)

    R2_spca = 1 - SSE_pca / SSE_ar

    print("R2_spca: ", R2_spca, "SSE_pca: ", SSE_pca, "SSE_AR: ", SSE_ar)

    return R2_spca


def main():
    variables = get_data()
    data = variables['data']
    inflation = variables['inflation']

    out_sample(y = inflation, X = data.values)

if __name__ == '__main__':
    main()
