"""
Class PredictNonlinear
"""

# Author: Joseph Park
# License: BSD 3 clause

from copy import copy
from multiprocessing import get_context
from itertools import repeat

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import Tags, InputTags, TargetTags
from pandas import DataFrame

from .smap import SMap
from .aux_func import ComputeError


class PredictNonlinear(TransformerMixin, BaseEstimator):
    """Evaluate nonlinearity (state-dependence) 

    This class is a wrapper for `SMap`. The goal is to estimate the optimal `SMap`
    localization parameter `theta`. 

    Both time series `columns` & `target` must be present as named columns in X.
    `columns` and `target` can be the same representing a univariate observation,
    or can be different in which case the variables are cross mapped with
    `columns` as the shadow manifold source and `target` the variable to predict.

    Parameters
    ----------
    columns : [str]
        Vector of column names to create embedding library

    target : str
        DataFrame column name of target feature to predict

    theta : [float]
        Vector of theta values, theta >= 0.

    lib : [int]
        Vector of pairs of 1-offset integer indices defining the embedding library
        The first index of a pair is the start index, the second the stop index.
        Default lib=None will assign lib=[1,N_obs] where N_obs is the number of
        observations. lib is _not_ 0-offset but ranges from 1 to N_obs.

    pred : [int]
        Vector of pairs of 1-offset integer indices defining target prediction times
        The first index of a pair is the start index, the second the stop index.
        Default pred=None will assign pred=[1,N_obs] where N_obs is the number of
        observations. pred is _not_ 0-offset but ranges from 1 to N_obs.

    tau : int
        Embedding time delay offset. Negative are delays, positive future values.

    Tp : int
        Prediction horizon in units of time series row indices

    exclusionRadius : int
        Temporal exclusion radius for nearest neighbors. Neighbors closer than
        exclusionRadius indices from the target are ignored. Not applicable if
        `lib` and `pred` are disjoint. 

    embedded : bool
        Is the input an embedding? If False (default) all columns will be time
        delay embedded with E and tau.

    noTime : bool
        X is expected to be DataFrame with time index/values/strings in first
        column. If no time vector is provided set noTime=True.

    mpMethod : str
        Multiprocessing context start method
        See: docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods


    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    E_rho_ : DataFrame
        DataFrame of embedding dimension and maximal simplex predictive correlation


    Returns
    -------


    Examples
    --------
    >>> from skedm import PredictNonlinear
    >>> from pandas import read_csv
    >>> df = read_csv('data/S12CD-S333-SumFlow_1980-2005.csv')
    >>> pnl = PredictNonlinear(columns='SumFlow', target='SumFlow', E=3,
                              lib=[1,700], pred=[701,1379], Tp=3)
    >>> thetaRho = pnl.fit_transform(df)
    >>> from aux_func import PlotPredictNonlinear
    >>> PlotPredictNonlinear(thetaRho, E=pnl.E, Tp=pnl.Tp)

    Notes
    -----

    See Also
    --------
    """

    # Used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {"no_validation"}

    def __init__(
        self,
        columns=None,
        target=None,
        theta=[0.01,0.1,0.3,0.5,0.75,1,1.5,2,3,4,5,6,7,8,9],
        E=1,
        lib=None,
        pred=None,
        Tp=1,
        tau=-1,
        exclusionRadius=0,
        embedded=False,
        noTime=False,
        mpMethod=None,
        chunksize=1,
        n_jobs=10,
    ):
        """Parameters are considered static;
        If mutable they should be copied before being modified.
        """
        self.columns = columns
        self.target = target
        self.theta = theta,
        self.E = E
        self.lib = lib
        self.pred = pred
        self.Tp = Tp
        self.tau = tau
        self.exclusionRadius = exclusionRadius
        self.embedded = embedded
        self.noTime = noTime
        self.mpMethod = mpMethod
        self.chunksize = chunksize
        self.n_jobs = n_jobs

    def __sklearn_tags__(self):
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            input_tags=InputTags(allow_nan=True),
        )

    def get_feature_names_out(self, input_features=None):
        """set_output for downstream pipeline compatibility

        The 'output' is a DataFrame with [E. rho]"""
        check_is_fitted(self)
        return array(["theta", "rho"], dtype=object)

    # -------------------------------------------------------------------
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """This method does no work. It copies mutable parameters and sets
        some feature objects for scikit-learn compatibility. It does not
        call `validate_data()` as it is called in `transform` and again in
        each `Simplex` object. 

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Observed data to be embedded, or used as embedding.

        y : accepted but silently ignored

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        """

        self._validate_params()

        # sklearn requires static __init__ parameters
        # instantiate copies that can be updated based on EDM or test criteria
        self._columns = copy(self.columns)
        self._target = copy(self.target)
        self._embedded = copy(self.embedded)
        self._noTime = copy(self.noTime)
        self._theta = copy(self.theta)[0] # list has been embedded in tuple wth?
        self._name = "PredictNonlinear"

        # Declare / instantiate class objects for understandability
        # Externally presented attributes
        self.theta_rho_ = None  # DataFrame

        # scikit-learn conventions
        self.feature_names_in_ = X.columns
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    # -------------------------------------------------------------------
    def transform(self, X):
        """Use multiprocessing pool to evaluate theta with a `SMap` predictor.

        A list of embedding dimensions `Evals` holds the `E` to be evaluated
        for simplex predictive fidelity. The list is used to create the
        `_poolArgs` iterable fed to a multiprocessing context which executes
        the `SimplexE()` function for each iterable item. 

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Passed to instances of class `Simplex`: see simplex.py

        Returns
        -------
        DataFrame
            [theta, rho]
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Set reset=False to not overwrite `n_features_in_` and
        # `feature_names_in_` but check the shape is consistent.
        X = validate_data(
            self, X=X, y=None, accept_sparse=False, skip_check_array=True, reset=False
        )

        # Setup Pool arguments
        args = {
            "columns": self._columns,
            "target": self._target,
            "E": self.E,
            "lib": self.lib,
            "pred": self.pred,
            "Tp": self.Tp,
            "tau": self.tau,
            "exclusionRadius": self.exclusionRadius,
            "embedded": self._embedded,
            "noTime": self._noTime,
        }

        # Create iterable for Pool.starmap, use repeated copies of data, args
        self._poolArgs = zip(self._theta, repeat(X), repeat(args))

        # Multiargument starmap : call self.SMapTheta()
        mpContext = get_context(self.mpMethod)
        with mpContext.Pool(processes=self.n_jobs) as pool:
            rhoList = pool.starmap(
                self.SMapTheta, self._poolArgs, chunksize=self.chunksize
            )

        self.theta_rho_ = DataFrame({"theta": self._theta, "rho": rhoList})

        return self.theta_rho_

    # -------------------------------------------------------------------
    def SMapTheta(self, theta, data, args):
        """SMap prediction with theta"""

        # Simplex instance
        S = SMap(
            columns=args["columns"],
            target=args["target"],
            E=args["E"],
            tau=args["tau"],
            Tp=args["Tp"],
            lib=args["lib"],
            pred=args["pred"],
            theta=theta,
            solver=None,
            exclusionRadius=args["exclusionRadius"],
            embedded=args["embedded"],
            noTime=args["noTime"],
        )

        _ = S.fit(data)
        p = S.predict(data)

        err = ComputeError(S.Projection_["Observations"], S.Projection_["Predictions"])
        return err["rho"]

    # -------------------------------------------------------------------
    def set_output(self, transform="pandas"):
        pass
