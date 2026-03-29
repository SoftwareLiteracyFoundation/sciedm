"""
============================
PredictNonlinear Transformer
============================

Examine nonlinearity through the `S-Map` `theta` parameter.

Parameter `theta` is the spatial scale factor of an exponential kernel in the state space (embedding) weighting nearest neighbors. The peak of predictability (`rho`), if not at `theta=0`, indicates nonlinear state dependence and an optimal value for `theta` in `SMap` predictions.
"""
from pandas import read_csv
from skedm import PredictNonlinear

df = read_csv("../skedm/data/S12CD-S333-SumFlow_1980-2005.csv")
pnl = PredictNonlinear(columns='SumFlow', target='SumFlow', E=4, Tp=3)
pnl.fit_transform(df)

# Plot
from skedm.aux_func import PlotPredictNonlinear
PlotPredictNonlinear(pnl.theta_rho_,
                     title=f"{pnl.columns} : {pnl.target}  E={pnl.E}  Tp={pnl.Tp}")


