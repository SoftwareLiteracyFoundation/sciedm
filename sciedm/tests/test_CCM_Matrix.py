"""EmbedDimension tests against pyEDM validation files."""


import pytest
from numpy import float16, float32, array_equal, round
from sciedm.datasets import load_dataset
from sciedm import CCM_Matrix

from .conftest import CCM_MatrixArgs, ValidData


def test_ccm_matrix_lorenz():
    '''CCM_Matrix on Lorenz5D V1'''
    data = load_dataset('Lorenz5D')
    kwargs = CCM_MatrixArgs.copy()
    kwargs.update( dict( E                = 5,
                         pLibSizes        = [10,20,80,90],
                         exclusionRadius  = 10,
                         sample           = 100,
                         seed             = 7777,
                         sharedMB         = 0.001 ) )
    cmat = CCM_Matrix( **kwargs )

    tensor, columns, libSizes = cmat.fit_transform( data )

    rhoDF_valid = ValidData('CCM_Matrix_Lorenz5D_rho_valid.csv')
    ra = rhoDF_valid.iloc[:,1:].round(2).to_numpy().astype(float16)

    slopeDF_valid = ValidData('CCM_Matrix_Lorenz5D_slope_valid.csv')
    sa = slopeDF_valid.iloc[:,1:].round(3).to_numpy().astype(float16)

    assert array_equal( ra, round( tensor[:,:,3], 2 ), equal_nan=True )
    assert array_equal( sa, round( cmat.slope_, 3 ), equal_nan=True )
