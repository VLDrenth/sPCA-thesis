import pandas as pd
import numpy as np
import os
from scipy.stats import t
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from src.helpers.functions import pc_T, R2_sklearn, winsor, get_data, select_AR_lag_SIC, estimate_AR_res
from src.helpers.regression import linear_reg


def in_sample(y, z):
    """ This function computes the in-sample R^2 and the in-sample adjusted R^2."""

    out = []
    out_table = []
    loadings = []
    horizon = [1]
    maxp = [2]
    kn = 15
    Zs = (z - np.mean(z, axis=0)) / np.std(z, axis=0)
    T = y.shape[0]

    for k in range(len(horizon)):
        h = horizon[k]

        y_h = np.zeros(T - (h - 1))
        for t in range(T - (h - 1)):
            y_h[t] = np.mean(y[t:t + h])
        
        p_max = maxp[k]
        p_AR_star_n = select_AR_lag_SIC(y, h, p_max)

        print('p_AR_star_n: ', p_AR_star_n)
        a_hat, res_h = estimate_AR_res(y, h, p_AR_star_n)

        beta = np.full(Zs.shape[1], np.nan)
        tstat = np.full(Zs.shape[1], np.nan)
            
        for j in range(Zs.shape[1]):
            #lm = lr.fit(Zs[:-h, j].reshape(-1, 1), y_h[1:])
            #parm = lm.coef_
            parm, std_err, t_stat, reg_se, adj_r2, bic = linear_reg(y_h[h:], Zs[:-h, j].reshape(-1, 1), constant=1, nlag=h)
            beta[j] = parm[1]

        # Winsorizing should be done at (0, 90)
        beta_win = winsor(np.abs(beta), p=(0, 90))
        scaleZs = np.zeros(Zs.shape)

        # Scale the factors by the winsorized betas
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
            parm, std_err, t_stat, reg_se, adj_r2_pc, bic = linear_reg(res_h, z_pc[p_AR_star_n-1:-h, :l+1], constant=1, nlag=h)
            adr2_pc[l] = adj_r2_pc

            parm, std_err, t_stat, reg_se, adj_r2_spc, bic = linear_reg(res_h, z_spc[p_AR_star_n-1:-h, :l+1], constant=1, nlag=h)
            adr2_spc[l] = adj_r2_spc
        
        out.append(np.vstack((adr2_pc, adr2_spc)).T * 100)
        out_table.append(np.vstack((var_explained_pc[:kn], var_explained_spc[:kn])).T)
    
    output = np.vstack(out)
    output = np.round(output, 2)

    return output

def main():
    variables = get_data()
    data = variables['data']
    #inflation = variables['inflation']
    unemployment = variables['unemployment']
    #ip_growth = variables['ip_growth']
    
    test = in_sample(y = unemployment, z = data.values)
    print(test)

if __name__ == '__main__':
    main()

