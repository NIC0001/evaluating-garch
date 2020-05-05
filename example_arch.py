import numpy as np
import pandas as pd
from arch import arch_model

if '__main__' == __name__:
    """
     
     Ref: https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/
    
    """
    # create data
    df = pd.read_csv('./dat/JP7203_from19830104to20200501.csv', index_col=0)
    returns = np.diff(np.log(df['adjusted'].values))
    
    # create model
    model = arch_model(returns, mean='Zero', vol='GARCH', p=1, q=1)
    x = model.fit()
    print(x.summary())

    """
                           Zero Mean - GARCH Model Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.000
    Mean Model:                 Zero Mean   Adj. R-squared:                  0.000
    Vol Model:                      GARCH   Log-Likelihood:                24449.5
    Distribution:                  Normal   AIC:                          -48893.0
    Method:            Maximum Likelihood   BIC:                          -48871.6
                                            No. Observations:                 9134
    Date:                Wed, May 06 2020   Df Residuals:                     9131
    Time:                        03:37:05   Df Model:                            3
                                  Volatility Model
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    omega      7.1520e-06  1.267e-09   5646.208      0.000 [7.150e-06,7.155e-06]
    alpha[1]       0.1000  2.382e-02      4.198  2.695e-05   [5.331e-02,  0.147]
    beta[1]        0.8800  1.661e-02     52.979      0.000     [  0.847,  0.913]
    ============================================================================
    
    Covariance estimator: robust
"""
