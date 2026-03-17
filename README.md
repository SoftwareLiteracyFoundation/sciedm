skedm - Empirical Dynamic Modeling with scikit-learn
============================================================

**skedm** provides Empirical Dynamic Modeling [EDM](https://en.wikipedia.org/wiki/Empirical_dynamic_modeling) tools within the [scikit-learn](https://scikit-learn.org) ecosystem.

#### Demo
Here we demonstrate `skedm.Simplex` and `OrthogonalMatchingPursuitCV` on a multivariate time series prediction task. For demonstration purposes a scikit-learn compatible Regressor is derived from `class(RegressorMixin, BaseEstimator)`.

A five variable nonlinear (chaotic) dynamical system is represented in `data/Lorenz5D.csv` from the Lorenz'96 model. 

```python
from pandas import read_csv
df = read_csv('Lorenz5D.csv') # see ./data/
df.head(3)
    Time      V1      V2      V3      V4      V5
0  10.00  2.4873  1.0490  3.4093  8.6502 -2.4232
1  10.05  3.5108  2.2832  4.0464  7.8964 -2.1931
2  10.10  4.1666  3.7791  4.7456  6.8123 -1.8866
```

The task is to predict variable `V2` from the other four. In EDM nomenclature the embedding is constructed from `columns`, we also assign the `target`:

```python
columns, target = ['V1','V3','V4','V5'], 'V2'

X,y = df[columns], df[target]
```

To specify a multivariate embedding instead of a time-delay embedding we set the parameter `embedded=True`:
```python
from skedm import Simplex
smp = Simplex(columns=columns, target=target, Tp=0, embedded=True)
smp.fit(df)
smp.score(df, y)
0.9406
```

`Simplex.predict()` creates a `Projection_` attribute, a DataFrame of the `Observations` (`target`) along with `Predictions`

```python
smp.Projection_.head(2)
    Time  Observations  Predictions  Pred_Variance
0  10.00        1.0490     1.122024       0.221640
1  10.05        2.2832     2.244541       0.431804
2  10.10        3.7791     3.825371       0.534411
```

We compare to OrthogonalMatchingPursuitCV:

```python
from sklearn.linear_model import OrthogonalMatchingPursuitCV
omp = OrthogonalMatchingPursuitCV(cv=5)
omp.fit(X, y)
ompPredictions = omp.predict(X)
omp.score(X, y)
0.29601
```
