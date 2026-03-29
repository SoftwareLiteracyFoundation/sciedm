"""
Class CCM : Convergent cross mapping of column : target
"""

# Author: Joseph Park
# License: BSD 3 clause

from copy import copy
from warnings import warn
from multiprocessing import get_context

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import Tags, InputTags, TargetTags, check_random_state
from numpy import array, divide, exp, fmax, nan, ndarray, mean, power
from numpy import roll, subtract, sum, zeros
from pandas import DataFrame, concat

from .simplex import Simplex
from .aux_func import ComputeError


class CCM(TransformerMixin, BaseEstimator):
    """Convergent cross mapping (CCM) of columns vs. target

    Both time series `columns` & `target` must be present as named columns in X.
    The direction of cross-mapping is set at construction via
    `columns` (shadow manifold source) and `target` (variable to predict).
    The reverse mapping is also computed.

    Parameters
    ----------
    columns : [str]
        Vector of column names to create embedding library

    target : str
        DataFrame column name of target feature to predict

    E : int
        Embedding dimension applied in time delay embedding

    libSizes : [int]
        Vector of library sizes to evaluate cross map convergence

    sample : int
        Number of random state vector shufflings at each libSize

    random_state : int
        Random number random_state for state vector shufflings

    tau : int
        Embedding time delay offset. Negative are delays, positive future values.

    Tp : int
        Prediction horizon in units of time series row indices

    knn : int
        Number of k-nearest neighbors for the simplex. If default knn=0 it is
        set to E+1.

    exclusionRadius : int
        Temporal exclusion radius for nearest neighbors. Neighbors closer than
        exclusionRadius indices from the target are ignored.

    embedded : bool
        Is the input an embedding? If False (default) all columns will be time
        delay embedded with E and tau.

    noTime : bool
        X is expected to be DataFrame with time index/values/strings in first
        column. If no time vector is provided set noTime=True.

    includeData : bool
        Populate the PredictStats1_ and PredictStats2_ attributes with DataFrames
        of the cross map correlation at each sample of each library size

    mpMethod : str
        Multiprocessing context start method
        See: docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

    sequential : bool
        If True do not use multiprocessing to evaluate forward and reverse mappings
        in parallel.

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    libMeans_ : DataFrame
        DataFrame with columns: ["Library Size", column:target, target:column]
        listing cross map predictive correlation at each library size

    PredictStats1_ : DataFrame
        DataFrame with cross map statistics at each library sample : FwdMap

    PredictStats2_ : DataFrame
        DataFrame with cross map statistics at each library sample : RevMap


    Returns
    -------


    Examples
    --------
    >>> from skedm import CCM
    >>> from pandas import DataFrame
    >>> df = DataFrame({'time':[t for t in range(1,21)],
                        'x':[1,1,3,4,5,5,6,8,3,3,2,6,5,5,9,3,5,1,8,2],
                        'y':[5,5,7,8,6,6,7,8,2,2,2,8,3,3,7,5,3,1,1,1]})
    >>> ccm = CCM(columns='x',target='y',E=2,libSizes=[10,12,16,18])
    >>> ccm.fit(df)
    CCM(E=2, columns='x', libSizes=[10, 12, 16, 18], target='y')
    >>> ccm.transform(df)

    Notes
    -----

    See Also
    --------

    Reference
    ---------
    Sugihara, George et al. (2012). Detecting Causality in Complex Ecosystems
    Science. 338 (6106): 496–500. doi:10.1126/science.1227079
    """

    # Used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {"no_validation"}

    def __init__(
        self,
        columns=None,
        target=None,
        E=1,
        libSizes=None,
        Tp=0,
        tau=-1,
        knn=0,
        sample=30,
        random_state=None,
        exclusionRadius=0,
        embedded=False,
        noTime=False,
        includeData=False,
        mpMethod=None,
        sequential=False,
    ):
        """Parameters are considered static;
        If mutable they should be copied before being modified.
        """
        self.columns = columns
        self.target = target
        self.E = E
        self.libSizes = libSizes
        self.Tp = Tp
        self.tau = tau
        self.knn = knn
        self.sample = sample
        self.random_state = random_state
        self.exclusionRadius = exclusionRadius
        self.embedded = embedded
        self.noTime = noTime
        self.includeData = includeData
        self.mpMethod = mpMethod
        self.sequential = sequential


    def __sklearn_tags__(self):
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            input_tags=InputTags(allow_nan=True),
        )

    def get_feature_names_out(self, input_features=None):
        """set_output for downstream pipeline compatibility
        
        The 'output' is a DataFrame with [LibSize, Col:Target, Target:Col]"""
        check_is_fitted(self)
        return array(["LibSize",
                      f"{self._columns}:{self._target}",
                      f"{self._target}:{self._columns}"], dtype=object)

    # -------------------------------------------------------------------
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Initialize lib & pred indices, embed data, find neighbors

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Observed data to be embedded, or used as embedding.

        y : accepted but silently ignored — embedding is target-independent

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        """

        # sklearn requires static __init__ parameters
        # instantiate copies that can be updated based on EDM or test criteria
        self._columns = copy(self.columns)
        self._target = copy(self.target)
        self._E = copy(self.E)
        self._knn = copy(self.knn)
        self._embedded = copy(self.embedded)
        self._noTime = copy(self.noTime)
        self._name = "CCM"

        # Declare / instantiate class objects for understandability
        # Externally presented attributes
        self.libMeans_ = None  # DataFrame CCM output
        self.PredictStats1_ = None  # DataFrame of CrossMap stats
        self.PredictStats2_ = None  # DataFrame of CrossMap stats

        # Internally utilized
        self._CrossMapList = None  # List of CrossMap results

        # Create two Simplex instances
        self.FwdMap = Simplex(
            columns=self._columns,
            target=self._target,
            E=self._E,
            tau=self.tau,
            Tp=self.Tp,
            lib=None,
            pred=None,
            knn=self._knn,
            exclusionRadius=self.exclusionRadius,
            embedded=self._embedded,
            noTime=self._noTime,
        )
        _ = self.FwdMap.fit(X)  # Wasted call to .FindNeighbors()

        self.RevMap = Simplex(
            columns=self._target,
            target=self._columns,
            E=self._E,
            tau=self.tau,
            Tp=self.Tp,
            lib=None,
            pred=None,
            knn=self._knn,
            exclusionRadius=self.exclusionRadius,
            embedded=self._embedded,
            noTime=self._noTime,
        )
        _ = self.RevMap.fit(X)  # Wasted call to .FindNeighbors()

        self.feature_names_in_ = X.columns
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    # -------------------------------------------------------------------
    def transform(self, X):
        """Call CrossMap() for forward and reverse mappings.

        The output is independent of the specific X passed here
        (CCM was fit on the training X); this method exists so
        the estimator participates correctly in sklearn Pipelines.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Not used but accepted for sklearn convention.

        Returns
        -------
        DataFrame
            CCM output: columns [LibSize, columns:target, target:columns]
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Set reset=False to not overwrite `n_features_in_` and
        # `feature_names_in_` but check the shape is consistent.
        X = validate_data(
            self, X=X, y=None, accept_sparse=False, skip_check_array=True, reset=False
        )

        if self.sequential:
            FwdCM = self.CrossMap("FWD")
            RevCM = self.CrossMap("REV")
            self._CrossMapList = [FwdCM, RevCM]
        else:
            # multiprocessing Pool CrossMap both directions simultaneously
            poolArgs = ["FWD", "REV"]
            mpContext = get_context(self.mpMethod)
            with mpContext.Pool(processes=2) as pool:
                CrossMapList = pool.map(self.CrossMap, poolArgs)

            self._CrossMapList = CrossMapList

        FwdCM, RevCM = self._CrossMapList

        self.libMeans_ = DataFrame(
            {
                "LibSize": FwdCM["libRho"].keys(),
                f"{FwdCM['columns'][0]}:{FwdCM['target']}": FwdCM["libRho"].values(),
                f"{RevCM['columns'][0]}:{RevCM['target']}": RevCM["libRho"].values(),
            }
        )

        if self.includeData:
            FwdCMStats = FwdCM["predictStats"]  # key libSize : list of CE dicts
            RevCMStats = RevCM["predictStats"]

            FwdCMDF = []
            for libSize in FwdCMStats.keys():
                LibSize = [libSize] * self.sample  # this libSize sample times
                libStats = FwdCMStats[libSize]  # sample ComputeError dicts

                libStatsDF = DataFrame(libStats)
                libSizeDF = DataFrame({"LibSize": LibSize})
                libDF = concat([libSizeDF, libStatsDF], axis=1)

                FwdCMDF.append(libDF)

            RevCMDF = []
            for libSize in RevCMStats.keys():
                LibSize = [libSize] * self.sample  # this libSize sample times
                libStats = RevCMStats[libSize]  # sample ComputeError dicts

                libStatsDF = DataFrame(libStats)
                libSizeDF = DataFrame({"LibSize": LibSize})
                libDF = concat([libSizeDF, libStatsDF], axis=1)

                RevCMDF.append(libDF)

            FwdStatDF = concat(FwdCMDF, axis=0)
            RevStatDF = concat(RevCMDF, axis=0)

            self.PredictStats1_ = FwdStatDF
            self.PredictStats2_ = RevStatDF

        return self.libMeans_  # DataFrame of mean cross map rho at each libSize

    # -------------------------------------------------------------------
    def CrossMap(self, direction):
        """Simplex cross mapping with loops for library sizes and samples"""
        if direction == "FWD":
            S = self.FwdMap
        elif direction == "REV":
            S = self.RevMap
        else:
            raise RuntimeError(f"{self._name}: CrossMap() Invalid Map")

        S._targetVec = S._Data[S._target].to_numpy()

        # Create random number generator : None sets random state from OS
        RNG = check_random_state(self.random_state)

        # Copy S.lib_i since it's replaced every iteration
        lib_i = S.lib_i_.copy()
        N_lib_i = len(lib_i)

        libRhoMap = {}  # Output dict libSize key : mean rho value
        libStatMap = {}  # Output dict libSize key : list of ComputeError dicts

        # ------------------------------------------------------------
        # Loop for library sizes
        # ------------------------------------------------------------
        for libSize in self.libSizes:
            rhos = zeros(self.sample)

            if self.includeData:
                predictStats = [None] * self.sample

            # --------------------------------------------------------
            # Loop for samples
            # --------------------------------------------------------
            for s in range(self.sample):
                # Generate library row indices for this sample
                # S.lib_i_ = RNG.choice( lib_i, size = min( libSize, N_lib_i ),
                #                       replace = False )
                S.lib_i_ = RNG.choice(lib_i, size=min(libSize, N_lib_i), replace=False)

                S.FindNeighbors()  # Depends on S.lib_i & S.knn

                # Code from Simplex:Project ---------------------------------
                # First column is minimum distance of all N pred rows
                minDistances = S.knn_distances_[:, 0]
                # In case there is 0 in minDistances: minWeight = 1E-6
                minDistances = fmax(minDistances, 1e-6)

                # Divide each column of N x k knn_distances by minDistances
                scaledDistances = divide(S.knn_distances_, minDistances[:, None])
                weights = exp(-scaledDistances)  # Npred x k
                weightRowSum = sum(weights, axis=1)  # Npred x 1

                # Matrix of knn_neighbors + Tp defines library target values
                knn_neighbors_Tp = S.knn_neighbors_ + S.Tp  # Npred x k
                libTargetValues  = S._targetVec[knn_neighbors_Tp]
                # Code from Simplex:Project ----------------------------------

                # Projection is average of weighted knn library target values
                projection_ = sum(weights * libTargetValues, axis=1) / weightRowSum

                # Align observations & predictions as in FormatProjection()
                # Shift projection_ by Tp
                if S.Tp != 0:
                    projection_ = roll(projection_, S.Tp)
                    if S.Tp > 0:
                        projection_[: S.Tp] = nan
                    elif S.Tp < 0:
                        projection_[S.Tp :] = nan

                err = ComputeError(S._targetVec[S.pred_i_], projection_, digits=5)

                rhos[s] = err["rho"]

                if self.includeData:
                    predictStats[s] = err

            libRhoMap[libSize] = mean(rhos)

            if self.includeData:
                libStatMap[libSize] = predictStats

        # Reset S.lib_i to original
        S.lib_i_ = lib_i

        if self.includeData:
            return {
                "columns": S._columns,
                "target": S._target,
                "libRho": libRhoMap,
                "predictStats": libStatMap,
            }
        else:
            return {"columns": S._columns, "target": S._target, "libRho": libRhoMap}

    # -------------------------------------------------------------------
    def set_output(self, transform="pandas"):
        pass
