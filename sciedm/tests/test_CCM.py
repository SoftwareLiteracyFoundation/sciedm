"""CCM tests against pyEDM validation files."""

import pytest
from numpy import nan
from sciedm.datasets import load_dataset
from sciedm import CCM

from .conftest import CCMArgs, ValidData


def test_ccm_anchovy_sst():
    '''sardine/anchovy/sst dataset'''
    data = load_dataset('sardine_anchovy_sst')
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'anchovy',
                         target   = 'np_sst',
                         libSizes = [10, 20, 30, 40, 50, 60, 70, 75],
                         sample   = 100,
                         E        = 3,
                         Tp       = 0,
                         tau      = -1,
                         random_state = 777 ) )
    ccm = CCM( **kwargs )
    df  = ccm.fit_transform( data )
    dfv = round( ValidData( 'CCM_anch_sst_valid.csv' ), 2 )

    assert dfv.equals( round( df, 2 ) )


def test_ccm_multivariate_lorenz():
    '''Multivariate columns'''
    data = load_dataset('Lorenz5D')
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'V3 V5',
                         target   = 'V1',
                         libSizes = [20, 200, 500, 950],
                         sample   = 30,
                         E        = 5,
                         Tp       = 10,
                         tau      = -5,
                         random_state = 777 ) )
    ccm = CCM( **kwargs )
    df  = ccm.fit_transform( data )
    dfv = round( ValidData( 'CCM_Lorenz5D_MV_valid.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )


def test_ccm_nan():
    '''nan in data'''
    data = load_dataset('circle').copy()
    data.iloc[ [5,  6, 12], 1 ] = nan
    data.iloc[ [10, 11, 17], 2 ] = nan

    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'x',
                         target   = 'y',
                         libSizes = [10, 190, 10],
                         sample   = 20,
                         E        = 2,
                         Tp       = 5,
                         tau      = -1,
                         random_state = 777 ) )
    ccm = CCM( **kwargs )
    df  = ccm.fit_transform( data )
    dfv = round( ValidData( 'CCM_nan_valid.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )


def test_ccm_negative_tp():
    '''Tp = -5'''
    data = load_dataset('circle')
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'x',
                         target   = 'y',
                         libSizes = [20, 200, 50],
                         sample   = 10,
                         E        = 2,
                         Tp       = -5,
                         tau      = -1,
                         random_state = 777 ) )
    ccm = CCM( **kwargs )
    df  = ccm.fit_transform( data )
    dfv = round( ValidData( 'CCM_NegativeTp.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )



def test_ccm_exclusion_radius():
    '''exclusionRadius = 5'''
    data = load_dataset('circle')
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns         = 'x',
                         target          = 'y',
                         libSizes        = [20, 200, 30],
                         sample          = 10,
                         E               = 2,
                         Tp              = 3,
                         tau             = -1,
                         exclusionRadius = 5,
                         random_state    = 777 ) )
    ccm = CCM( **kwargs )
    df  = ccm.fit_transform( data )
    dfv = round( ValidData( 'CCM_exclusionRadius.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )


def test_ccm_positive_tau():
    '''tau = +3 (positive)'''
    data = load_dataset('circle')
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'x',
                         target   = 'y',
                         libSizes = [20, 200, 30],
                         sample   = 10,
                         E        = 2,
                         Tp       = 0,
                         tau      = 3,
                         random_state = 777 ) )
    ccm = CCM( **kwargs )
    df  = ccm.fit_transform( data )
    dfv = round( ValidData( 'CCM_positiveTau.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )
