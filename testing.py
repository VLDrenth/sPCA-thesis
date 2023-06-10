import numpy as np
import pandas as pd
from src.helpers.refactored import loocv_ts
from src.helpers.functions import get_data

vars = get_data()
X = vars["data"].values
y = vars["inflation"].values

loocv_ts(X, y, 1, 1, 'pca')
