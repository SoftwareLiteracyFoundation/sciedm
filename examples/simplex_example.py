"""
===========================
Simplex Estimator
===========================

`Simplex` prediction of Lorenz'96 variables.

Four use cases:
  Out of sample prediction of V3 from time delay embedding of V3
  Out of sample prediction of V3 from time delay embedding of V1 (cross mapping)
  Out of sample prediction of V3 from multivariate embedding of V1,V2,V4,V5
  Out of sample prediction of V3 from mixed-multivariate embedding of V1,V2,V4,V5
"""
from pandas import read_csv
from matplotlib import pyplot as plt

from skedm import Simplex

df = read_csv("../skedm/data/Lorenz5D.csv")

# library (training) and prediction (test) sets: unit-offset row number [start,stop]
lib, pred = [1,500], [801,900]

# Out of sample prediction V3 from time delay embedding of V3
smp1 = Simplex(columns='V3', target='V3', E=5, lib=lib, pred=pred)
smp1.fit(df)
rho1 = smp1.score(df, df['V3'])

# Out of sample prediction of V3 from time delay embedding of V1 (cross mapping)
smp2 = Simplex(columns='V1', target='V3', E=5, lib=lib, pred=pred)
smp2.fit(df)
rho2 = smp2.score(df, df['V3'])

# Out of sample prediction of V3 from multivariate embedding of V1,V2,V4,V5
smp3 = Simplex(columns=['V1','V2','V4','V5'], target='V3', embedded=True,
               lib=lib, pred=pred)
smp3.fit(df)
rho3 = smp3.score(df, df['V3'])

# Out of sample prediction of V3 from mixed-multivariate embedding of V1,V2,V4,V5
smp4 = Simplex(columns=['V1','V2','V4','V5'], target='V3', E=2, lib=lib, pred=pred)
smp4.fit(df)
rho4 = smp4.score(df, df['V3'])

# Plot
from skedm.aux_func import PlotObsPred
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8.5,6))
(ax1, ax2, ax3, ax4) = (ax[0,0], ax[0,1], ax[1,0], ax[1,1])

title = f"{smp1.columns} : {smp1.target} rho={rho1:.2f} E={smp1._E} Tp={smp1.Tp}"
_ax1 = PlotObsPred(smp1.Projection_, title, ax=ax1)

title = f"{smp2.columns} : {smp2.target} rho={rho2:.2f} E={smp2._E} Tp={smp2.Tp}"
_ax2 = PlotObsPred(smp2.Projection_, title, ax=ax2)

title = f"{smp3.columns} : {smp3.target} rho={rho3:.2f} E={smp3._E} Tp={smp3.Tp}"
_ax3 = PlotObsPred(smp3.Projection_, title, ax=ax3)

title = f"{smp4.columns} : {smp4.target} rho={rho4:.2f} E={smp4._E} Tp={smp4.Tp}"
_ax4 = PlotObsPred(smp4.Projection_, title, ax=ax4)

plt.tight_layout()
plt.show()
