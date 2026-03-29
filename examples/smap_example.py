"""
===========================
SMap Estimator
===========================

`SMap` prediction of Lorenz'96 variables.

Two use cases:
  Out of sample prediction of V3 from time delay embedding of V3
  Out of sample prediction of V3 from multivariate embedding of V1,V2,V4,V5
"""
from pandas import read_csv
from matplotlib import pyplot as plt

from skedm import SMap

df = read_csv("../skedm/data/Lorenz5D.csv")

# library (training) and prediction (test) sets: unit-offset row number [start,stop]
lib, pred = [1,500], [801,900]

# Out of sample prediction V3 from time delay embedding of V3
smp1 = SMap(columns='V3', target='V3', E=5, lib=lib, pred=pred, theta=3)
smp1.fit(df)
rho1 = smp1.score(df, df['V3'])

# Out of sample prediction of V3 from multivariate embedding of V1,V2,V4,V5
smp2 = SMap(columns=['V1','V2','V4','V5'], target='V3', embedded=True,
            lib=lib, pred=pred, theta=3)
smp2.fit(df)
rho2 = smp2.score(df, df['V3'])

# Plot
from skedm.aux_func import PlotObsPred, PlotCoeff
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8.5,6))
(ax1, ax2, ax3, ax4) = (ax[0,0], ax[0,1], ax[1,0], ax[1,1])

title = f"{smp1.columns} : {smp1.target} rho={rho1:.2f} E={smp1._E} Tp={smp1.Tp}"
_ax1 = PlotObsPred(smp1.Projection_, title, ax=ax1)

title = f"{smp2.columns} : {smp2.target} rho={rho2:.2f} E={smp2._E} Tp={smp2.Tp}"
_ax2 = PlotObsPred(smp2.Projection_, title, ax=ax2)

# SMap coefficient of V3 vs previous time 
ax3.plot(smp1.Coefficients_['Time'], smp1.Coefficients_['∂V3/∂V3(t-1)'],
         lw=3, c='forestgreen', label='∂V3/∂V3(t-1)')
ax3.legend()

# SMap coefficient of V3 vs V1
ax4.plot(smp2.Coefficients_['Time'], smp2.Coefficients_['∂V3/∂V1'],
         lw=3, c='darkred', label='∂V3/∂V1')
ax4.legend()

plt.tight_layout()
plt.show()
