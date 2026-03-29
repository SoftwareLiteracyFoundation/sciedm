"""
=============================
CCM Transformer
=============================

Compute convergent cross mapping (CCM) between two components of Lorenz'96 5-D system.

`libSizes` specify the set of library sizes evaluated for convergence.
"""
from pandas import read_csv
from skedm import CCM

df = read_csv("../skedm/data/Lorenz5D.csv")
ccm1 = CCM(columns='V1', target='V5', E=7, libSizes=[25,50,100,200,800,900,950])
ccm1.fit_transform(X=df)

from skedm.aux_func import PlotCCM
PlotCCM(ccm1.libMeans_, title=f"E={ccm1.E} {ccm1.columns} : {ccm1.target}")
