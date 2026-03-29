"""
===========================
EmbedDimension Transformer
===========================

Estimate embedding dimension (`E`) of Everglades flow data.

The estimate of `E` corresponds to the first peak in predictability (`rho`) or the point of saturation of predictability. 
"""
from pandas import read_csv
from skedm import EmbedDimension

df = read_csv("../skedm/data/S12CD-S333-SumFlow_1980-2005.csv")

edim = EmbedDimension(columns='SumFlow', target='SumFlow', exclusionRadius=3)
E_rho = edim.fit_transform(df)

# Plot
from skedm.aux_func import PlotEmbedDimension
PlotEmbedDimension(edim.E_rho_, title=f"{edim.columns} : {edim.target}  Tp={edim.Tp}")
