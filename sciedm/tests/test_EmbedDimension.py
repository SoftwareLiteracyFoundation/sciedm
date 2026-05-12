"""EmbedDimension tests against pyEDM validation files."""
import warnings # Remove > python 3.13 : Pickle, copy, and deepcopy ... itertools

import pytest
from numpy import nan
from sciedm.datasets import load_dataset
from sciedm import EmbedDimension

from .conftest import EmbedDimensionArgs, ValidData


def test_embed_dimension_lorenz():
    '''EmbedDimension on Lorenz5D V1'''
    data = load_dataset('Lorenz5D')
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update( dict( columns         = 'V1',
                         target          = 'V1',
                         maxE            = 12,
                         lib             = [1, 500],
                         pred            = [501, 800],
                         Tp              = 15,
                         tau             = -5,
                         exclusionRadius = 20,
                         n_jobs          = 10 ) )
    edim = EmbedDimension( **kwargs )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _ = edim.fit_transform( data )

    df   = edim.E_rho_
    dfv  = round( ValidData( 'EmbedDim_valid.csv' ), 6 )

    assert dfv.equals( round( df, 6 ) )
