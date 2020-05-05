import numpy as np
import pyflux as pf
import pandas as pd
from datetime import datetime

if '__main__' == __name__:
    """
     Ref: https://pyflux.readthedocs.io/en/latest/garch.html
    
    """
    df = pd.read_csv('./dat/JP7203_from19830104to20200501.csv', index_col=0)
    
    returns = np.diff(np.log(df['adjusted'].values))
    
    model = pf.GARCH(returns,p=1,q=1)
    x = model.fit()
    x.summary()

    """
        GARCH(1,1)
    ======================================================= ==================================================
    Dependent Variable: Series                              Method: MLE
    Start Date: 1                                           Log Likelihood: 24478.6392
    End Date: 9133                                          AIC: -48949.2785
    Number of observations: 9133                            BIC: -48920.7999
    ==========================================================================================================
    Latent Variable                          Estimate   Std Error  z        P>|z|    95% C.I.
    ======================================== ========== ========== ======== ======== =========================
    Vol Constant                             0.0
    q(1)                                     0.1882
    p(1)                                     0.7872
    Returns Constant                         0.0005     0.0095     0.0508   0.9595   (-0.0182 | 0.0192)
    ==========================================================================================================
"""