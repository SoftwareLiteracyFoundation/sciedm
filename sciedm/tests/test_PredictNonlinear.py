"""EmbedDimension tests against pyEDM validation files."""
import warnings # Remove > python 3.13 : Pickle, copy, and deepcopy ... itertools

import pytest
from numpy import nan
from sciedm.datasets import load_dataset
from sciedm import PredictNonlinear

from .conftest import PredictNonlinearArgs, ValidData


def test_predict_nonlinear_tentmap():
    '''PredictNonlinear on TentMapNoise'''
    data = load_dataset('TentMapNoise')
    kwargs = PredictNonlinearArgs.copy()
    kwargs.update( dict( columns    = 'TentMap',
                         target     = 'TentMap',
                         lib        = [1, 500],
                         pred       = [501, 800],
                         E          = 4,
                         Tp         = 1,
                         tau        = -1,
                         n_jobs     = 10,
                         theta      = [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5,
                                       2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
                        )
                  )
    nl = PredictNonlinear( **kwargs )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _ = nl.fit_transform( data )

    df   = nl.theta_rho_
    dfv  = round( ValidData( 'PredictNonlinear_valid.csv' ), 6 )

    assert dfv.equals( round( df, 6 ) )
