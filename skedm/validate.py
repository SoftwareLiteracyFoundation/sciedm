from warnings import warn
from copy import copy
from datetime import datetime

from sklearn.utils.validation import validate_data
from scipy.sparse import issparse
from pandas import DataFrame, Series
from numpy import any, array, isinf, isnan, ndarray

from .aux_func import IsIterable


# skedm.simplex .smap method
def Validate(self):
    """Validate DataFrame columns/target in .fit()"""

    # Ensure columns are provided, convert to iterable if bare string
    if self._columns is None:
        raise RuntimeError(f"Validate() {self._name}: columns required")

    if not IsIterable(self._columns):
        self._columns = [self._columns]

    for column in self._columns:
        if column not in self._Data.columns:
            raise RuntimeError(
                f"Validate() {self._name}: column "
                + f"{column} not found in dataFrame."
            )

    if self._target is None:
        raise RuntimeError(f"Validate() {self._name}: target required")

    if self._target not in self._Data.columns:
        raise RuntimeError(
            f"Validate() {self._name}: target "
            + f"{self._target} not found in dataFrame."
        )


# skedm.simplex .smap method
def Validate_Xy(self, X, y):
    # validate_data() validates input data and sets or checks feature
    # names and counts of the input. This mutates the estimator setting
    # n_features_in_ and feature_names_in_ attributes if reset=True.
    # Enable skip_check_array to allow pandas instead of ndarray coercion
    # Returns {ndarray, sparse matrix} or tuple of these
    Z = validate_data(
        self,
        X=X,
        y=y,
        reset=True,
        dtype="numeric",
        validate_separately=False,
        skip_check_array=True,
        accept_sparse=False,
        ensure_all_finite="allow-nan",
        y_numeric=True,
    )

    if isinstance(Z, tuple):
        _X, _y = X, y
    else:
        _X, _y = X, None

    # A pandas DataFrame in X per pyEDM is expected, set to self._Data
    # If X is not DataFrame create self._Data with _columns= ['x1','x2',..]
    # _target='y' and issue warning.
    if isinstance(_X, DataFrame):
        self._Data = _X.copy()  # DataFrame passed in : presume user API call

    elif issparse(_X):
        msg = f"{self._name} sparse input not supported; got {type(_X).__name__}"
        raise TypeError(msg)

    elif isinstance(_X, list) or isinstance(_X, ndarray):
        warn(
            f"{self._name} received ndarray but column names are load-bearing "
            "metadata for embedding and target selection. "
            "Provide a pandas DataFrame with column names and a time vector "
            "in the first column for application.",
            UserWarning,
            stacklevel=2,
        )
        if isinstance(_X, list):
            _X = array(_X)
            if isinstance(_y, list):
                _y = array(_y)

        if self._name == "SMap":
            self._embedded = True
            warn(
                f"{self._name} received multivariate ndarray. Setting embedded=True",
                UserWarning,
                stacklevel=2,
            )

        # Create columns names for X as ['x1','x2',..]
        if _X.ndim == 1:
            num_cols = _X.shape[0]
        elif _X.ndim == 2:
            num_cols = _X.shape[1]
        self._columns = getattr(
            self, "feature_names_in_", array([f"x{i+1}" for i in range(num_cols)])
        )
        self._Data = DataFrame(_X, columns=self._columns)
        if _y is not None and len(_y):
            self._target = "y"
            self._Data.insert(self._Data.shape[1], self._target, _y)

        # Set self._noTime True to add time vector in EDM_params()
        self._noTime = True

    else:
        msg = (
            f"{self._name} requires DataFrame or ndarray; " + f"got {type(X).__name__}"
        )
        raise ValueError(msg)

    if self._Data.empty:
        msg = f"{self._name} DataFrame X is empty."
        raise ValueError(msg)

    if self._Data.shape[0] < 11:
        msg = f"{self._name} skedm requires at least 10 observations."
        raise ValueError(msg)

    if not len(self._columns):
        msg = (
            f"{self._name} 0 feature(s) (shape={_X.shape}) "
            + "while a minimum of 1 is required."
        )
        raise ValueError(msg)

    if isinstance(_y, ndarray) and (any(isinf(_y)) or all(isnan(_y))):
        msg = f"{self._name} inf not allowed in target"
        raise ValueError(msg)


# skedm.simplex .smap method
def RemoveNan(self):
    """KDTree in Neighbors does not accept nan
    If ignoreNan remove Embedding rows with nan from lib_i, pred_i"""
    if self._ignoreNan:
        # type : <class 'pandas.core.series.Series'>
        # Series of bool for all Embedding columns (axis = 1) of lib_i...
        na_lib = self.Embedding_.iloc[self.lib_i_, :].isna().any(axis=1)
        na_pred = self.Embedding_.iloc[self.pred_i_, :].isna().any(axis=1)

        if na_lib.any():
            if self._name == "SMap":
                original_knn = self._knn
                original_lib_i_len = len(self.lib_i_)

            # Redefine lib_i excluding nan
            self.lib_i_ = self.lib_i_[~na_lib.to_numpy()]

            # lib_i resized, update SMap self.knn if not user provided
            if self._name == "SMap":
                if original_knn == original_lib_i_len - 1:
                    self._knn = len(self.lib_i_) - 1

        # Redefine pred_i excluding nan
        if any(na_pred):
            self.pred_i_ = self.pred_i_[~na_pred]

    self.PredictionValid()


# skedm.simplex .smap method
def PredictionValid(self):
    """Validate there are pred_i to make a prediction"""
    if len(self.pred_i_) == 0:
        msg = (
            f"{self._name}: PredictionValid() No valid prediction "
            + "indices. Examine pred, E, tau, Tp parameters and/or nan."
        )
        warn(msg)
