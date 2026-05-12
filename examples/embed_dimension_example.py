"""
===========================
EmbedDimension Transformer
===========================

Estimate embedding dimension (`E`) of Everglades flow data.

The estimate of `E` corresponds to the first peak in predictability (`rho`) or the point of saturation of predictability. 
"""
from pandas import read_csv
from sciedm import EmbedDimension
from sciedm.datasets import load_dataset

df = load_dataset("SumFlow")

edim = EmbedDimension(columns='SumFlow', target='SumFlow', exclusionRadius=3)
E_rho = edim.fit_transform(df)

# Plot
from sciedm.aux_func import PlotEmbedDimension
PlotEmbedDimension(edim.E_rho_, title=f"{edim.columns} : {edim.target}  Tp={edim.Tp}")
