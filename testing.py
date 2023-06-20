import numpy as np
import pandas as pd
from src.helpers.refactored import loocv_ts
from src.helpers.functions import get_data

for nfac in [1, 2, 3]:
    for method in ["PCA", "sPCA"]:
        for heterosked in [True, False]:
            for n in [10, 20, 30, 40, 50]:
                print("nfac: {}, method: {}, heterosked: {}, n: {}".format(nfac, method, heterosked, n))
                # Save errors in npy file
                errors =  np.load("resources/results/sim/errors_nfac_{}_method_{}_heterosked_{}_n_{}.npy".format(nfac, method, heterosked, n))
                #print median mse
                mse_vec = (errors**2).mean(axis=1)
                print("Median MSE: {}".format(np.median(mse_vec)))

