"""
Class CCM_Matrix : Convergent cross mapping of all column vs all column

For M data columns, computes CCM correlation of each column against
every other column at each library size, returning an (M, M, L)
float16 tensor with a convergence depth dimension.

Optimization: for a fixed source column i the KDTree construction,
neighbor query, self/exclusion filtering, and simplex weight computation
depend only on column i's embedding — not on the target. The expensive
per-subsample work is done once and the resulting weights are reused
across all M-1 target columns via vectorized batched prediction.

Post-processing:
 - Linear convergence slope: vectorized numpy regression across the
   full M×M matrix simultaneously, producing an (M, M) float32 array.
 - Exponential convergence (optional): scipy curve_fit of
   rho(L) = y0 + b*(1 - exp(-a*x)) per cell, storing the rate
   parameter `a` as an (M, M) float32 array.

Reference: George Sugihara et al., Detecting Causality in Complex Ecosystems.
           Science338, 496-500(2012). DOI:10.1126/science.1227079
"""

# Author: Joseph Park
# License: BSD 3 clause

# Python distribution modules
from datetime        import datetime
from time            import perf_counter
from multiprocessing import cpu_count, shared_memory
from multiprocessing import get_context, get_start_method
import sys
import warnings

# Community modules
from scipy.spatial  import KDTree
from scipy.optimize import curve_fit
from matplotlib     import pyplot as plt
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import Tags, InputTags, TargetTags, check_random_state


# Worker process global
_mw_data = {}


class CCM_Matrix(TransformerMixin, BaseEstimator):
    """
    Compute the full M×M×L convergent cross mapping tensor.

    Parameters
    ----------
    E : int or array-like of int
        Embedding dimension. Scalar or per-column vector of length M.
    Tp : int
        Prediction horizon. Default 0.
    tau : int
        Embedding delay. Default -1.
    exclusionRadius : int
        Temporal exclusion radius. Default 0.
    libSizes : list of int
        Explicit library sizes. If non-empty, used directly and
        pLibSizes is ignored. Default [].
    pLibSizes : list of float
        Percentiles of N to generate library sizes. Used only when
        libSizes is empty. Default [10, 20, 80, 90].
    sample : int
        Subsamples per library size. Default 100.
    seed : int or None
        RNG seed.
    noTime : bool
        If True, all columns are data. If False, first column is
        time (stripped). Default False.
    parallel : bool or int
        Worker count. Default True.
    mpMethod : str or None
        Multiprocessing start method: 'forkserver' or 'spawn' only.
        Default None.
    sharedMB : float
        Data size threshold for shared memory vs pickle. Default 5.
    targetBatchSize : int or None
        Max target columns per batch within each worker. Default None.
    expConverge : bool
        If True, fit exponential convergence curve. Default False.
    progressLog : None, True, or str
        None: no logging. True: log to stderr. str: log to file path.
        Default None.
    progressInterval : int
        Percentage increment for progress log lines. Default 5.

    Attributes (after Run)
    --------------------------
    tensor_ : ndarray (M, M, L), float16
    slope_ : ndarray (M, M), float16
    exp_a_ : ndarray (M, M), float32 or None
    columns_ : list of str
    lib_sizes_arr_ : ndarray of int
    lib_sizes_norm_ : ndarray of float

    The slope of CCM rho(libSizes) is computed based on a [0,1]
    normalization of libSizes.

    if expConverge = True a nonlinear convergence function is fit
    to rho(libSizes) : y0 + b * ( 1 - exp(-a * x) ) with fit coefficient
    a returned in the (M, M) matrix self.exp_a
    
    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.


    Returns
    -------
    tuple (MxMxL tensor, columns)


    Examples
    --------
    >>> from sciedm import CCM_Matrix, PlotMatrix
    >>> from pandas import DataFrame
    >>> X = np.random.rand(10,10)
    >>> df = DataFrame(X,columns=[f"x{i+1}" for i in range(10)])
    >>> cmat = CCM_Matrix(E=4,noTime=True)
    >>> tensor,columns = cmat.fit_transform(df)
    >>> PlotMatrix(tensor[:,:,2],columns)

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

    _validated = False # Internal state flags
    _UNSET = object() # If called outside API can override args

    def __init__(
        self,
        E,
        libSizes         = [],
        pLibSizes        = [10,20,80,90],
        Tp               = 0,
        tau              = -1,
        exclusionRadius  = 0,
        sample           = 30,
        seed             = None,
        noTime           = False,
        parallel         = True,
        mpMethod         = None,
        sharedMB         = 0.01,
        targetBatchSize  = None,
        expConverge      = False,
        progressLog      = None,
        progressInterval = 5
    ):
        """Parameters are considered static;
        If mutable they should be copied before being modified.
        """
        self.E                = E
        self.Tp               = Tp
        self.tau              = tau
        self.exclusionRadius  = exclusionRadius
        self.libSizes         = libSizes
        self.pLibSizes        = pLibSizes
        self.sample           = sample
        self.seed             = seed
        self.noTime           = noTime
        self.parallel         = parallel
        self.mpMethod         = mpMethod
        self.sharedMB         = sharedMB
        self.targetBatchSize  = targetBatchSize
        self.expConverge      = expConverge
        self.progressLog      = progressLog
        self.progressInterval = progressInterval


    def __sklearn_tags__(self):
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            input_tags=InputTags(allow_nan=True),
        )

    def get_feature_names_out(self, input_features=None):
        """set_output for downstream pipeline compatibility

        The 'output' is a MxMxL tensor with M columns_ names"""
        check_is_fitted(self)
        return np.array(self.columns_, dtype=object,)

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
        self._dataFrame = X
        self._name = "CCM_Matrix"

        # Externally presented attributes
        self.columns_        = None
        self.tensor_         = None
        self.slope_          = None
        self.exp_a_          = None
        self.lib_sizes_arr_  = None
        self.lib_sizes_norm_ = None

        self.Validate()

        # scikit-learn compliance
        self.feature_names_in_ = self.columns_
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
            tuple: MxMxL tensor, columns
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Set reset=False to not overwrite `n_features_in_` and
        # `feature_names_in_` but check the shape is consistent.
        X = validate_data(
            self, X=X, y=None, accept_sparse=False,
            skip_check_array=True, reset=True
        )

        self.Run()

        return self.tensor_, self.columns_


    # ================================================================
    # Validate
    # ================================================================

    def Validate(self):
        """Parse DataFrame, extract data, resolve E and library sizes."""
        df = self._dataFrame

        if self.noTime:
            self.columns_ = list(df.columns)
            self.data_matrix = np.ascontiguousarray(
                df.values.astype(np.float64)
            )
        else:
            self.columns_ = list(df.columns[1:]) # exclude 1st column time
            self.data_matrix = np.ascontiguousarray(
                df.iloc[:, 1:].values.astype(np.float64)
            )

        self.N, self.M = self.data_matrix.shape

        E_input = self.E
        if np.ndim(E_input) == 0:
            # scalar E : create M dimensional vector
            self.E_vec = np.full(self.M, int(E_input), dtype=int)
        else:
            self.E_vec = np.asarray(E_input, dtype=int)
            if len(self.E_vec) != self.M:
                raise ValueError(
                    f"E has length {len(self.E_vec)} but there are "
                    f"{self.M} data columns."
                )

        self.k_vec = self.E_vec + 1

        if len(self.libSizes) > 0:
            self.lib_sizes_arr_ = np.asarray(
                self.libSizes, dtype=int
            )
        else:
            pcts = np.asarray(self.pLibSizes, dtype=float)
            self.lib_sizes_arr_ = np.unique(np.clip(
                np.round(pcts / 100.0 * self.N).astype(int),
                2, self.N
            ))
            self.libSizes = self.lib_sizes_arr_

        self.lib_sizes_norm_ = (self.lib_sizes_arr_.astype(np.float64)
                                / self.N)

        if self.M < 2:
            raise ValueError(
                f"Need at least 2 data columns, got {self.M}."
            )
        if np.any(self.E_vec < 1):
            raise ValueError("All E values must be >= 1.")

        self._validated = True


    # ================================================================
    # Run
    # ================================================================

    def Run(self):
        """
        Compute the M×M×L CCM tensor, linear convergence slope,
        and (optionally) exponential convergence rate.

        Returns
        -------
        tensor : ndarray (M, M, |L|), float16
        """
        if not self._validated:
            self.Validate()

        M = self.M
        N = self.N
        n_lib     = len(self.lib_sizes_arr_)
        logging   = self.progressLog is not None
        dest      = self.progressLog # None, True or filename
        interval  = self.progressInterval
        n_workers = _resolve_workers(self.parallel)

        if logging:
            datetime_start = datetime.now()
            _log_progress(dest, "CCM_Matrix.Run() starting.")

        root_seq = np.random.SeedSequence(self.seed)
        child_seqs = root_seq.spawn(M)

        tasks = []
        for i in range(M):
            tasks.append((
                i, M, N, int(self.E_vec[i]), self.tau,
                self.lib_sizes_arr_.tolist(),
                self.sample, self.exclusionRadius, self.Tp,
                self.targetBatchSize, child_seqs[i].entropy
            ))

        # ---- Dispatch ----
        if n_workers > 1:
            ctx = _resolve_mp_context(self.mpMethod)
            total_bytes = self.data_matrix.nbytes
            use_shm = total_bytes > self.sharedMB * 1_000_000

            if logging:
                results = self._dispatch_parallel_logged(
                    ctx, tasks, use_shm, M, n_workers, dest, interval
                )
            else:
                results = self._dispatch_parallel_silent(
                    ctx, tasks, use_shm, n_workers
                )
        else:
            _mw_data['data'] = self.data_matrix
            if logging:
                results = self._dispatch_sequential_logged(
                    tasks, M, dest, interval
                )
            else:
                results = [_mw_task(*t) for t in tasks]

        # ---- Assemble tensor ----
        tensor_f32 = np.full((M, M, n_lib), np.nan, dtype=np.float32)
        for src_idx, row in results:
            tensor_f32[src_idx, :, :] = row

        self.tensor_ = tensor_f32.astype(np.float16)

        if logging:
            msg = (f"100% ({M}/{M}) complete")
            _log_progress(dest, msg)

        # ---- Post-processing: convergence estimates ----
        t_post = perf_counter()

        if self.tensor_.shape[2] > 1:
            self.slope_ = _compute_slope(tensor_f32, self.lib_sizes_norm_)

        t_slope = perf_counter() - t_post
        datetime_end = datetime.now()

        if self.expConverge:
            t_exp0 = perf_counter()
            self.exp_a_ = _compute_exp_converge(
                tensor_f32, self.lib_sizes_norm_
            )
            t_exp = perf_counter() - t_exp0
            datetime_end = datetime.now()
        else:
            self.exp_a = None
            t_exp = 0.0

        if logging:
            msg = (f"slope complete: {_fmt_duration(t_slope)}")
            _log_progress(dest, msg)

            if self.expConverge:
                msg = f"expConverge: {_fmt_duration(t_exp)}"
                _log_progress(dest, msg)

            msg = f"Elapsed: {datetime_end - datetime_start}"
            _log_progress(dest, msg)

        return self.tensor_

    # ---- Parallel dispatch with logging ----

    def _dispatch_parallel_logged(self, ctx, tasks, use_shm,
                                  M, n_workers, dest, interval):
        chunksize = max(1, M // (4 * n_workers))
        results = []

        if use_shm:
            shm, spec = _create_shared_array(self.data_matrix)
            init_fn = _mw_init_shm
            init_args = ([spec],)
        else:
            shm = None
            init_fn = _mw_init_pickle
            init_args = (self.data_matrix,)

        try:
            with ctx.Pool(
                processes=n_workers,
                initializer=init_fn,
                initargs=init_args,
            ) as pool:
                completed = 0
                next_threshold = interval
                t_start = perf_counter()

                for result in pool.imap_unordered(
                    _mw_task_unpack, tasks, chunksize=chunksize
                ):
                    results.append(result)
                    completed += 1
                    pct = completed * 100.0 / M

                    if pct >= next_threshold:
                        elapsed = perf_counter() - t_start
                        rate = completed / elapsed
                        remaining = (M - completed) / rate
                        _log_progress(
                            dest,
                            f"{int(pct)}% ({completed}/{M}) | "
                            f"elapsed {_fmt_duration(elapsed)} | "
                            f"~{_fmt_duration(remaining)} remaining | "
                            f"{rate:.1f} tasks/s"
                        )
                        next_threshold += interval
        finally:
            if shm is not None:
                shm.close()
                shm.unlink()

        return results

    # ---- Parallel dispatch without logging ----

    def _dispatch_parallel_silent(self, ctx, tasks, use_shm, n_workers):
        chunksize = max(1, len(tasks) // (4 * n_workers))

        if use_shm:
            shm, spec = _create_shared_array(self.data_matrix)
            try:
                with ctx.Pool(
                    processes=n_workers,
                    initializer=_mw_init_shm,
                    initargs=([spec],),
                ) as pool:
                    results = pool.starmap(
                        _mw_task, tasks, chunksize=chunksize
                    )
            finally:
                shm.close()
                shm.unlink()
        else:
            with ctx.Pool(
                processes=n_workers,
                initializer=_mw_init_pickle,
                initargs=(self.data_matrix,),
            ) as pool:
                results = pool.starmap(
                    _mw_task, tasks, chunksize=chunksize
                )

        return results

    # ---- Sequential dispatch with logging ----

    def _dispatch_sequential_logged(self, tasks, M, dest, interval):
        results = []
        completed = 0
        next_threshold = interval
        t_start = perf_counter()

        for t in tasks:
            results.append(_mw_task(*t))
            completed += 1
            pct = completed * 100.0 / M

            if pct >= next_threshold:
                elapsed = perf_counter() - t_start
                rate = completed / elapsed
                remaining = (M - completed) / rate
                _log_progress(
                    dest,
                    f"{int(pct)}% ({completed}/{M}) | "
                    f"elapsed {_fmt_duration(elapsed)} | "
                    f"~{_fmt_duration(remaining)} remaining | "
                    f"{rate:.1f} tasks/s"
                )
                next_threshold += interval

        return results

    # -------------------------------------------------------------------
    #def set_output(self, transform="pandas"):
    #    pass


# ====================================================================
# Non class functions
# ====================================================================


# ====================================================================
# Batched Pearson correlation across columns
# ====================================================================

def _batched_pearson_cols(preds, actuals, skip_col, col_offset):
    """Pearson r per column between preds and actuals."""
    B = preds.shape[1]
    has_nan = np.isnan(preds).any() or np.isnan(actuals).any()

    if not has_nan:
        pm = preds - preds.mean(axis=0, keepdims=True)
        am = actuals - actuals.mean(axis=0, keepdims=True)
        num = np.sum(pm * am, axis=0)
        den = np.sqrt(np.sum(pm * pm, axis=0) * np.sum(am * am, axis=0))
        rhos = np.where(den > 0, num / den, 0.0).astype(np.float32)
    else:
        rhos = np.full(B, np.nan, dtype=np.float32)
        for b in range(B):
            j = col_offset + b
            if j == skip_col:
                continue
            rhos[b] = _nan_safe_pearson(preds[:, b], actuals[:, b])

    diag_b = skip_col - col_offset
    if 0 <= diag_b < B:
        rhos[diag_b] = np.nan

    return rhos


def _nan_safe_pearson(predictions, actuals):
    """Pearson r excluding NaN pairs. NaN if < 3 valid pairs."""
    valid = ~(np.isnan(predictions) | np.isnan(actuals))
    n = valid.sum()
    if n < 3:
        return np.nan
    p = predictions[valid]
    a = actuals[valid]
    pm = p - p.mean()
    am = a - a.mean()
    denom = np.sqrt(np.dot(pm, pm) * np.dot(am, am))
    if denom == 0.0:
        return 0.0
    return np.dot(pm, am) / denom


# ====================================================================
# pool and shared memory functions
# ====================================================================

def _mw_init_pickle(data_matrix):
    """Store data matrix passed via initargs (pickle path)."""
    _mw_data['data'] = data_matrix


def _mw_init_shm(shm_specs):
    """Attach to shared memory data matrix in worker process."""
    arrays = []
    handles = []
    for name, shape, dtype_str in shm_specs:
        shm = shared_memory.SharedMemory(name=name, create=False)
        handles.append(shm)
        arrays.append(
            np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        )
    _mw_data['data'] = arrays[0]
    _mw_data['_shm'] = handles


def _mw_task(src_idx, M, N, E_src, tau, lib_sizes, sample,
             exclusionRadius, Tp, target_batch_size, seed_entropy):
    """
    Process one source column → one row of the M×M×L tensor.
    """
    data = _mw_data['data']
    n_lib = len(lib_sizes)
    row = np.full((M, n_lib), np.nan, dtype=np.float32)

    k = E_src + 1
    shifts = np.arange(E_src) * tau

    src_vec = data[:, src_idx]
    embed_src, valid_src = _build_embedding(src_vec, shifts)
    valid_src &= ~np.isnan(src_vec)

    idx_src = np.where(valid_src)[0]
    M_valid = len(idx_src)

    if M_valid < E_src + 2:
        return src_idx, row

    embed_valid = np.ascontiguousarray(embed_src[idx_src])

    shifted_idx = idx_src + Tp
    in_bounds = (shifted_idx >= 0) & (shifted_idx < N)
    shifted_safe = np.clip(shifted_idx, 0, N - 1)

    target_all = data[shifted_safe, :].copy()
    target_all[~in_bounds, :] = np.nan

    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    r_Mv = np.arange(M_valid)[:, np.newaxis]

    if target_batch_size is None or target_batch_size <= 0:
        target_batch_size = M

    k_base = min(k, M_valid - 1)
    if k_base < 1:
        return src_idx, row

    for li, L in enumerate(lib_sizes):
        L = min(int(L), M_valid)
        k_use = min(k, L - 1)
        if k_use < 1:
            continue

        k_query = min(k_use + 1 + (2 * exclusionRadius
                                    if exclusionRadius > 0 else 0), L)

        sample_rhos = np.full((sample, M), np.nan, dtype=np.float32)

        for s in range(sample):
            lib_idx = rng.choice(M_valid, size=L, replace=False)
            lib_idx.sort()

            tree = KDTree(embed_valid[lib_idx])
            nn_dist_raw, nn_local_raw = tree.query(embed_valid, k=k_query)

            if nn_dist_raw.ndim == 1:
                nn_dist_raw  = nn_dist_raw[:, np.newaxis]
                nn_local_raw = nn_local_raw[:, np.newaxis]

            nn_global_raw = lib_idx[nn_local_raw]

            is_self = (nn_global_raw == r_Mv)

            if exclusionRadius > 0:
                src_rows = idx_src[:, np.newaxis]
                nn_rows  = idx_src[nn_global_raw]
                mask = is_self | (np.abs(src_rows - nn_rows)
                                  <= exclusionRadius)
            else:
                mask = is_self

            valid_nn = ~mask
            cs = np.cumsum(valid_nn, axis=1)
            first_k = valid_nn & (cs <= k_use)

            total_found = cs[:, -1]
            insufficient = total_found < k_use

            if np.all(insufficient):
                continue

            if np.any(insufficient):
                first_k[insufficient, :] = False

            _, col_indices = np.where(first_k)

            nn_cols = np.zeros((M_valid, k_use), dtype=np.intp)
            sufficient = ~insufficient
            suf_count = sufficient.sum()
            if suf_count > 0:
                nn_cols[sufficient] = col_indices.reshape(suf_count, k_use)

            nn_dist   = nn_dist_raw[r_Mv, nn_cols]
            nn_global = nn_global_raw[r_Mv, nn_cols]

            d_min    = nn_dist[:, 0:1]
            d_min_nz = np.where(d_min > 0.0, d_min, 1.0)
            weights  = np.exp(-nn_dist / d_min_nz)

            zero_mask = (d_min == 0.0)
            if np.any(zero_mask):
                weights = np.where(
                    zero_mask,
                    np.where(nn_dist == 0.0, 1.0, 0.0),
                    weights
                )
            w_sum = weights.sum(axis=1, keepdims=True)
            w_sum = np.where(w_sum > 0.0, w_sum, 1.0)
            weights /= w_sum

            if np.any(insufficient):
                weights[insufficient, :] = 0.0

            weights_3d = weights[:, :, np.newaxis]

            for t_start in range(0, M, target_batch_size):
                t_end = min(t_start + target_batch_size, M)
                tgt_batch = target_all[:, t_start:t_end]

                nn_tgt = tgt_batch[nn_global]
                preds = np.sum(weights_3d * nn_tgt, axis=1)

                if np.any(insufficient):
                    preds[insufficient, :] = np.nan

                batch_rhos = _batched_pearson_cols(
                    preds, tgt_batch, src_idx, t_start
                )
                sample_rhos[s, t_start:t_end] = batch_rhos

        any_valid = ~np.all(np.isnan(sample_rhos), axis=0)
        if np.any(any_valid):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                row[any_valid, li] = np.nanmean(
                    sample_rhos[:, any_valid], axis=0
                )

    row[src_idx, :] = np.nan
    return src_idx, row


def _mw_task_unpack(args):
    """Unpack single tuple argument for imap_unordered."""
    return _mw_task(*args)


# ====================================================================
# Convergence fitting functions
# ====================================================================

def _compute_slope(tensor, lib_sizes_norm):
    """
    Vectorized linear regression of CCM rho vs normalised library size
    across the full M×M matrix simultaneously.
    """
    M = tensor.shape[0]
    n_L = tensor.shape[2]

    Y = tensor.astype(np.float32).reshape(M * M, n_L)
    X = lib_sizes_norm.astype(np.float32)

    all_nan = np.all(np.isnan(Y), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        row_mean = np.nanmean(Y, axis=1, keepdims=True)
    row_mean = np.where(np.isnan(row_mean), 0.0, row_mean)
    Y = np.where(np.isnan(Y), row_mean, Y)

    x_bar = X.mean()
    x_dev = X - x_bar
    y_bar = Y.mean(axis=1, keepdims=True)
    y_dev = Y - y_bar

    ss_xy = y_dev @ x_dev
    ss_xx = np.dot(x_dev, x_dev)

    slope = np.where(ss_xx > 0, ss_xy / ss_xx, 0.0)
    slope[all_nan] = np.nan

    return slope.reshape(M, M).astype(np.float16)


def _CCM_rho_L_fit(x, a, b, y0):
    """CCM rho(L) curve for L normalised to [0,1]."""
    return (y0 + b * (1.0 - np.exp(-a * x))).flatten()


def _compute_exp_converge(tensor, lib_sizes_norm):
    """
    Fit rho(L) = y0 + b*(1 - exp(-a*x)) per cell.
    Returns only the rate parameter `a`.
    """
    from scipy.optimize import curve_fit

    M = tensor.shape[0]
    a_matrix = np.full((M, M), np.nan, dtype=np.float32)
    xdata = lib_sizes_norm.astype(np.float64)

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            ydata = tensor[i, j, :].astype(np.float64)
            if np.all(np.isnan(ydata)) or np.sum(np.isfinite(ydata)) < 3:
                continue
            try:
                popt, _ = curve_fit(
                    _CCM_rho_L_fit,
                    xdata  = xdata,
                    ydata  = ydata,
                    p0     = [2.0, 1.0, 0.1],
                    bounds = ([0, 0, 0], [100, 1, 1]),
                    method = 'dogbox',
                )
                a_matrix[i, j] = popt[0]
            except Exception:
                pass

    return a_matrix


# ====================================================================
# Progress logging
# ====================================================================

def _log_progress(dest, message):
    """
    Write a timestamped log line. Self-contained: opens and closes
    the file on each call so nothing is lost on hard termination.

    Parameters
    ----------
    dest : True (stderr), str (file path), or None (no-op)
    message : str
    """
    if dest is None:
        return

    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"

    if dest is True:
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass
    elif isinstance(dest, str):
        try:
            with open(dest, 'a') as f:
                f.write(line)
        except Exception:
            pass


def _fmt_duration(seconds):
    """Format seconds into 'Xh Ym Zs' or 'Ym Zs' or 'Zs'."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"

    
# ====================================================================
# Helpers
# ====================================================================

def _resolve_mp_context(mpMethod):
    """Return a multiprocessing context restricted to forkserver/spawn."""
    allowed = ('forkserver', 'spawn')
    if mpMethod is not None:
        if mpMethod not in allowed:
            raise ValueError(
                f"mpMethod must be one of {allowed}, got '{mpMethod}'."
            )
        return get_context(mpMethod)
    default = get_start_method(allow_none=True)
    if default in allowed:
        return get_context(default)
    return get_context('forkserver')


def _resolve_workers(parallel):
    """Resolve the parallel parameter to a worker count."""
    if parallel is True:
        return max(1, cpu_count())
    if parallel is False:
        return 1
    n_workers = min(cpu_count(), int(parallel))
    return max(1, n_workers)


def _create_shared_array(arr):
    """Copy a numpy array into a new shared memory segment."""
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr
    return shm, (shm.name, arr.shape, arr.dtype.str)


def _build_embedding(vec, shifts):
    """Delay-coordinate embedding. Invalid rows retain NaN."""
    N = len(vec)
    E = len(shifts)
    emb = np.full((N, E), np.nan, dtype=np.float64)
    for dim, s in enumerate(shifts):
        if s <= 0:
            emb[-s:, dim] = vec[:N + s]
        else:
            emb[:N - s, dim] = vec[s:]
    valid = ~np.any(np.isnan(emb), axis=1)
    return emb, valid


# ====================================================================
# PlotMatrix
# ====================================================================

def PlotMatrix( xm, columns, figsize = (5,5), dpi = 150, title = None,
                plot = True, plotFile = None, cmap = None, norm = None,
                aspect = None, vmin = None, vmax = None, colorBarShrink = 1. ):
    '''Generic function to plot numpy matrix'''

    fig = plt.figure( figsize = figsize, dpi = dpi )
    ax  = fig.add_subplot()

    #fig.suptitle( title )
    ax.set( title = f'{title}' )
    ax.xaxis.set_ticks( [x for x in range( len(columns) )] )
    ax.yaxis.set_ticks( [x for x in range( len(columns) )] )
    ax.set_xticklabels(columns, rotation = 90)
    ax.set_yticklabels(columns)

    cax = ax.matshow( xm, cmap = cmap, norm = norm,
                      aspect = aspect, vmin = vmin, vmax = vmax )
    fig.colorbar( cax, shrink = colorBarShrink )

    plt.tight_layout()

    if plotFile :
        fname = f'{plotFile}'
        plt.savefig( fname, dpi = 'figure', format = 'png' )

    if plot :
        plt.show()
