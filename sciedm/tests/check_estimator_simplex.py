# sklearn canonical check_estimator()
import warnings
from sciedm import Simplex
from sklearn.utils.estimator_checks import check_estimator

expected_failed_checks = {
    "check_complex_data":"KDTree does not work with complex data",
    "check_regressor_data_not_an_array":"sciedm requires DataFrame or ndarray",
    "check_supervised_y_2d":"sciedm is not supervised, target is 1d",
    "check_dtype_object":"raw object dtype not allowed",
    "check_methods_sample_order_invariance":"sciedm is dynamical",
    "check_methods_subset_invariance":"sciedm is state dependent",
    "check_fit2d_1sample":"sciedm is dynamical, multiple observations required",
    "check_dict_unchanged":"sciedm requires numeric data",
    "check_fit2d_predict1d":"sciedm correctly creates DataFrame for this case",
    "check_fit2d_1feature":"sciedm requires at least 10 observations, 1 feature OK",
    "check_regressors_no_decision_function":"sciedm requires at least 10 observations",
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    L = check_estimator(Simplex(), expected_failed_checks=expected_failed_checks)
