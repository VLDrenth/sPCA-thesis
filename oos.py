import pandas as pd
import numpy as np
import os
from scipy.stats import t
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from src.helpers.functions import pc_T, R2_sklearn, winsor

def select_AR_lag_SIC(y, h, p_max):
    """
    Selects the optimal lag length for the AR model using the SIC.
    """
    T = len(y)
    AIC = np.zeros(p_max)
    for p in range(p_max):
        a_hat, res = estimate_AR_res(y, h, p)
        sigma2 = np.sum(res ** 2) / (T - p - 1)
        AIC[p] = np.log(sigma2) + 2 * (p + 1) / T
    p_star = np.argmin(AIC)
    return p_star

def estimate_AR_res(y, h, p):
    """
    Estimates the AR model and returns the residuals.
    """
    T = len(y)
    X = np.zeros((T - p, p + 1))
    for i in range(T - p):
        X[i, :] = np.concatenate((np.array([1]), y[i:i + p]))
    y_h = y[p:]
    lm = LinearRegression()
    lm.fit(X, y_h)
    a_hat = lm.coef_
    res = y_h - lm.predict(X)
    return a_hat, res

def in_sample(y, z):
    """ This function computes the in-sample R^2 and the in-sample adjusted R^2."""

    out = []
    out_table = []
    loadings = []
    horizon = [1]
    maxp = [3]
    kn = 15

    lr = LinearRegression()
    pca = PCA(n_components=kn)

    for k in range(len(horizon)):
        h = horizon[k]
        Zs = (z - np.mean(z, axis=0)) / np.std(z, axis=0)

        T = y.shape[0]
        y_h = np.zeros(T - (h - 1))
        for t in range(T - (h - 1)):
            y_h[t] = np.mean(y[t:t + h])
        
        p_max = maxp[k]
        p_AR_star_n = select_AR_lag_SIC(y, h, p_max)
        a_hat, res_h = estimate_AR_res(y, h, p_AR_star_n)

        beta = np.full(Zs.shape[1], np.nan)
        tstat = np.full(Zs.shape[1], np.nan)
            
        for j in range(Zs.shape[1]):
            lm = lr.fit(Zs[:-h, j].reshape(-1, 1), y_h[1:])
            parm = lm.coef_
            beta[j] = parm[0]

        
        beta_win = winsor(np.abs(beta), p=(0, 100))
        print(beta_win)
        scaleZs = np.zeros(Zs.shape)
        for j in range(Zs.shape[1]):
            scaleZs[:, j] = Zs[:, j] * beta_win[j]

        pca = PCA(n_components=kn)
        
        _, z_pc, _, _, _ = pc_T(Zs, kn)
        _, z_spc, _, _, _ = pc_T(scaleZs, kn)


        pca.fit(Zs)
        loadings_pc, var_explained_pc = pca.transform(Zs), pca.explained_variance_ratio_
        
        pca.fit(scaleZs)
        loadings_spc, var_explained_spc = pca.transform(scaleZs), pca.explained_variance_ratio_

        if h == 1:
            loadings.append(loadings_pc)
            loadings.append(loadings_spc)
            loadings.append(beta_win)
        
        z_pc = (z_pc - np.mean(z_pc, axis=0)) / np.std(z_pc, axis=0)
        z_spc = (z_spc - np.mean(z_spc, axis=0)) / np.std(z_spc, axis=0)

        adr2_pc = np.full(kn, np.nan)
        adr2_spc = np.full(kn, np.nan)

        for l in range(kn):
            lm_pc = lr.fit(z_pc[p_AR_star_n-1:-h, :l+1], res_h)
            adr2_pc[l] = R2_sklearn(lm_pc, z_pc[p_AR_star_n-1:-h, :l+1], res_h)

            lm_spc = lr.fit(z_spc[p_AR_star_n-1:-h, :l+1], res_h)
            adr2_spc[l] = R2_sklearn(lm_spc, z_spc[p_AR_star_n-1:-h, :l+1], res_h)
        

        out.append(np.vstack((adr2_pc, adr2_spc)).T * 100)
        out_table.append(np.vstack((var_explained_pc[:kn], var_explained_spc[:kn])).T)
    
    output = np.vstack(out)
    output = np.round(output, 2)

    return output
    
def main():
    # Load data
    file_path_clean = os.path.join(os.path.dirname(__file__), 'resources/data/data_fred_matlab.csv')
    file_path_raw = os.path.join(os.path.dirname(__file__), 'resources/data/raw_data_no_missing.csv')
    data = pd.read_csv(file_path_clean)
    raw_data = pd.read_csv(file_path_raw)

    # Set date as index of df
    data['sasdate'] = pd.to_datetime(data['sasdate'])
    data.set_index('sasdate', inplace=True)

    # Drop last column (unnamed)
    data.drop(data.columns[-1], axis=1, inplace=True)

    raw_data['sasdate'] = pd.to_datetime(raw_data['sasdate'])
    raw_data.set_index('sasdate', inplace=True)

    # Get the to be predicted variable
    inflation = np.log(raw_data['CPIAUCSL']).diff().dropna() * 100

    # Select only data from 1960-01-01 untill 2019-12-01
    data = data.loc[(data.index >= '1960-01-01') & (data.index <= '2019-12-01')]
    
    # Drop first few rows of raw data to match dimensions by taking last 720 rows
    inflation = inflation.iloc[-720:]

    print("Shapes of data and inflation: ", data.shape, inflation.shape)

    # Drop columns that are not used in original paper
    to_drop = ["ACOGNO", "TWEXAFEGSMTHx", "OILPRICEx", "VXOCLSx", "UMCSENTx"]
    data.drop(to_drop, axis=1, inplace=True)

    test = in_sample(y = inflation, z = data.values)
    print(test)

if __name__ == '__main__':
    main()

