
import sys
import unittest
from   datetime import datetime
import warnings # DeprecationWarning: multiprocessing in Python 3.13.

from numpy  import nan, array, array_equal
from pandas import DataFrame, read_csv
import sciedm

#----------------------------------------------------------------
# Suite of tests
#----------------------------------------------------------------
class test_sciedm( unittest.TestCase ):
    '''The examples/ should also run

    NOTE: Bizarre default of unittest class presumes
          methods names to be run begin with "test_" 
    '''
    # JP How to pass command line arg to class? verbose = True
    def __init__( self, *args, **kwargs):
        super( test_sciedm, self ).__init__( *args, **kwargs )

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    @classmethod
    def setUpClass( self ):
        self.verbose = False
        self.GetValidFiles( self )

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    def GetValidFiles( self ):
        '''Create dictionary of DataFrame values from file name keys'''
        self.ValidFiles = {}

        validFiles = [ 'CCM_anch_sst_valid.csv',
                       'CCM_Lorenz5D_MV_Space_valid.csv',
                       'CCM_Lorenz5D_MV_valid.csv',
                       'CCM_nan_valid.csv',
                       'EmbedDim_valid.csv',
                       'Multiview_combos_valid.csv',
                       'Multiview_pred_valid.csv',
                       'PredictInterval_valid.csv',
                       'PredictNonlinear_valid.csv',
                       'SMap_circle_E2_embd_valid.csv',
                       'SMap_circle_E4_valid.csv',
                       'SMap_nan_valid.csv',
                       'SMap_noTime_valid.csv',
                       'Smplx_DateTime_valid.csv',
                       'Smplx_disjointLib_valid.csv',
                       'Smplx_disjointPred_nan_valid.csv',
                       'Smplx_E3_block_3sp_valid.csv',
                       'Smplx_E3_embd_block_3sp_valid.csv',
                       'Smplx_exclRadius_valid.csv',
                       'Smplx_nan2_valid.csv',
                       'Smplx_nan_valid.csv',
                       'Smplx_negTp_block_3sp_valid.csv',
                       'Smplx_validLib_valid.csv' ]

        # Create map of module validFiles pathnames in validFiles
        for file in validFiles:
            filename = "validation/" + file
            self.ValidFiles[ file ] = read_csv( filename )

        # Get data files
        self.Lorenz5D = read_csv('../data/Lorenz5D.csv')
        self.SumFlow = read_csv('../data/S12CD-S333-SumFlow_1980-2005.csv')
        self.block3sp = read_csv('../data/block_3sp.csv')
        self.circle = read_csv('../data/circle.csv')
        self.circle_noTime = read_csv('../data/circle_noTime.csv')
        self.sardine_anchovy_sst = read_csv('../data/sardine_anchovy_sst.csv')
        self.TentMapNoise = read_csv('../data/TentMapNoise.csv')

    #------------------------------------------------------------
    # Simplex
    #------------------------------------------------------------
    def test_Simplex_1( self ):
        '''Simplex 1'''
        if self.verbose : print ( " --- Simplex 1 ---" )
        df_ = self.Lorenz5D
        spx = sciedm.Simplex(columns = 'V1', target = 'V5',
                             lib = [1,300], pred = [301,310], E = 5)
        spx.fit(df_)
        rho = spx.score(df_, df_['V5'])

    def test_Simplex_2( self ):
        '''Simplex 2'''
        if self.verbose : print ( "--- Simplex 2 ---" )
        df_ = self.Lorenz5D
        spx = sciedm.Simplex(columns = ['V1'], target = 'V5',
                             lib = [1, 300], pred = [301, 310], E = 5)
        spx.fit(df_)
        rho = spx.score(df_, df_['V5'])

    def test_Simplex_3( self ):
        '''Simplex 3'''
        if self.verbose : print ( "--- Simplex 3 ---" )
        df_ = self.Lorenz5D
        spx = sciedm.Simplex(columns = ['V1','V3'], target = 'V5',
                             lib = [1, 300], pred = [301, 310], E = 5)
        spx.fit(df_)
        rho = spx.score(df_, df_['V5'])

    def test_Simplex_4( self ):
        '''Simplex 5'''
        if self.verbose : print ( "--- Simplex 4 ---" )
        df_ = self.Lorenz5D
        spx = sciedm.Simplex(columns = 'V1', target = 'V5',
                             lib = [1, 300], pred = [301, 310], E = 5, knn = 0)
        spx.fit(df_)
        rho = spx.score(df_, df_['V5'])

    def test_Simplex_5( self ):
        '''Simplex 6'''
        if self.verbose : print ( "--- Simplex 5 ---" )
        df_ = self.Lorenz5D
        spx = sciedm.Simplex(columns = 'V1', target = 'V5',
                             lib = [1, 300], pred = [301, 310], E = 5, tau = -2)
        spx.fit(df_)
        rho = spx.score(df_, df_['V5'])

    def test_Simplex_6( self ):
        '''embedded = False'''
        if self.verbose : print ( "--- Simplex 6 embedded = False ---" )
        df_ = self.block3sp
        spx = sciedm.Simplex(columns = "x_t", target = "x_t",
                             lib=[1, 100], pred=[101, 195], E=3,
                             Tp=1, knn=0, tau=-1)
        spx.fit(df_)
        pred = spx.predict(df_)
        dfv = self.ValidFiles["Smplx_E3_block_3sp_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:95], 6 ) # Skip row 0 Nan
        S2 = round( spx.Projection_.get('Predictions')[1:95], 6 )# Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    def test_Simplex_7( self ):
        '''embedded = True'''
        if self.verbose : print ( "--- Simplex 7 embedded = True ---" )
        df_ = self.block3sp
        spx = sciedm.Simplex(columns = ["x_t", "y_t", "z_t"], target = "x_t",
                             lib=[1, 99], pred=[100, 198], E=3, Tp=1,
                             embedded = True)
        spx.fit(df_)
        pred = spx.predict(df_)
        dfv = self.ValidFiles["Smplx_E3_embd_block_3sp_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:98], 6 ) # Skip row 0 Nan
        S2 = round( spx.Projection_.get('Predictions')[1:98], 6 )# Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    def test_Simplex_8( self ):
        '''negative Tp'''
        if self.verbose : print ( "--- Simplex 8 negative Tp ---" )
        df_ = self.block3sp
        spx = sciedm.Simplex(columns = "x_t", target = "y_t",
                             lib=[1, 100], pred=[50, 80], E=3, Tp=-2)
        spx.fit(df_)
        pred = spx.predict(df_)
        dfv = self.ValidFiles["Smplx_negTp_block_3sp_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:98], 6 ) # Skip row 0 Nan
        S2 = round( spx.Projection_.get('Predictions')[1:98], 6 )# Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    # def test_Simplex_9( self ):
    #     '''validLib'''
    #     if self.verbose : print ( "--- Simplex 9 validLib ---" )
    #     df_ = self.circle
    #     df = sciedm.Simplex( dataFrame = df_, columns = 'x', target = 'x',
    #                       lib = [1,200], pred = [1,200], E = 2, Tp = 1,
    #                       validLib = df_.eval('x > 0.5 | x < -0.5') )

    #     dfv = self.ValidFiles["Smplx_validLib_valid.csv"]

    #     S1 = round( dfv.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
    #     S2 = round(  df.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
    #     self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_Simplex_10( self ):
        '''disjoint lib'''
        if self.verbose : print ( "--- Simplex 10 disjoint lib ---" )
        df_ = self.circle
        spx = sciedm.Simplex(columns = 'x', target = 'x',
                             lib = [1,40, 50,130], pred = [80,170],
                             E = 2, Tp = 1, tau = -3)
        spx.fit(df_)
        pred = spx.predict(df_)
        dfv = self.ValidFiles["Smplx_disjointLib_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        S2 = round( spx.Projection_.get('Predictions')[1:195],6 )# Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_Simplex_11( self ):
        '''disjoint pred w/ nan'''
        if self.verbose : print ( "--- Simplex 11 disjoint pred w/ nan ---" )
        df_ = self.Lorenz5D
        df_.iloc[ [8,50,501], [1,2] ] = nan

        spx = sciedm.Simplex(columns='V1', target = 'V2',
                             E = 5, Tp = 2, lib = [1,50,101,200,251,500],
                             pred = [1,10,151,155,551,555,881,885,991,1000])
        spx.fit(df_)
        pred = spx.predict(df_)
        dfv = self.ValidFiles["Smplx_disjointPred_nan_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:195], 5 ) # Skip row 0 Nan
        S2 = round( spx.Projection_.get('Predictions')[1:195],5 )# Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_Simplex_12( self ):
        '''exclusion radius'''
        if self.verbose : print ( "--- Simplex 12 exclusion radius ---" )
        df_ = self.circle
        spx = sciedm.Simplex(columns = 'x', target = 'y',
                             lib = [1,100], pred = [21,81], E = 2, Tp = 1,
                             exclusionRadius = 5)
        spx.fit(df_)
        pred = spx.predict(df_)
        dfv = self.ValidFiles["Smplx_exclRadius_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:60], 6 ) # Skip row 0 Nan
        S2 = round( spx.Projection_.get('Predictions')[1:60],6 )# Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_Simplex_13( self ):
        '''nan'''
        if self.verbose : print ( "--- Simplex 13 nan ---" )
        df_ = self.circle
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        spx = sciedm.Simplex(columns = 'x', target = 'y',
                             lib = [1,100], pred = [1,95], E = 2, Tp = 1 )
        spx.fit(dfn)
        pred = spx.predict(dfn)
        rho = spx.score(dfn, dfn['y'])
        dfv = self.ValidFiles["Smplx_nan_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:90], 6 ) # Skip row 0 Nan
        S2 = round( spx.Projection_.get('Predictions')[1:90], 6 )# Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_Simplex_14( self ):
        '''nan'''
        if self.verbose : print ( "--- Simplex 14 nan ---" )
        df_ = self.circle
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        spx = sciedm.Simplex(columns = 'y', target = 'x',
                             lib = [1,200], pred = [1,195], E = 2, Tp = 1)
        spx.fit(dfn)
        pred = spx.predict(df_)
        rho = spx.score(dfn, dfn['x'])
        dfv = self.ValidFiles["Smplx_nan2_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:190], 6 ) # Skip row 0 Nan
        S2 = round( spx.Projection_.get('Predictions')[1:190],6 )# Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_Simplex_15( self ):
        '''DateTime'''
        if self.verbose : print ( "--- Simplex 15 DateTime ---" )
        df_ = self.SumFlow

        spx = sciedm.Simplex(columns = 'SumFlow', target = 'SumFlow',
                             lib = [1,800], pred = [801,1001], E = 3, Tp = 1)
        spx.fit(df_)
        pred = spx.predict(df_)
        # self.assertTrue( isinstance( df_['Time'][0], datetime ) )

        dfv = self.ValidFiles["Smplx_DateTime_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:200], 6 ) # Skip row 0 Nan
        S2 = round( spx.Projection_.get('Predictions')[1:200],6 )# Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_Simplex_16( self ):
        '''knn = 1'''
        if self.verbose : print ( "--- Simplex 16 knn = 1 ---" )
        df_ = self.Lorenz5D

        spx = sciedm.Simplex(columns='V5', target = 'V5',
                             lib = [301,400], pred = [350,355],
                             knn = 1, embedded = True)
        spx.fit(df_)
        pred = spx.predict(df_)

        knn = spx.knn_neighbors_
        knnValid = array( [322,334,362,387,356,355] )[:,None]
        self.assertTrue( array_equal( knn, knnValid ) )

    #------------------------------------------------------------
    def test_Simplex_17( self ):
        '''exclusion Radius '''
        if self.verbose : print ( "--- Simplex 17 exclusion Radius ---" )
        df_ = self.Lorenz5D
        x   = [i+1 for i in range(1000)]
        df_ = DataFrame({'Time':df_['Time'],'X':x,'V1':df_['V1']})

        spx = sciedm.Simplex(columns='X', target = 'V1',
                             lib = [1,100], pred = [101,110],
                             E = 5, exclusionRadius = 10)
        spx.fit(df_)
        pred = spx.predict(df_)

        knn = spx.knn_neighbors_[:,0]
        knnValid = array( [89, 90, 91, 92, 93, 94, 95, 96, 97, 98] )
        self.assertTrue( array_equal( knn, knnValid ) )

    #------------------------------------------------------------
    # S-map
    #------------------------------------------------------------
    def test_SMap_1( self ):
        '''SMap'''
        if self.verbose : print ( "--- SMap 1 ---" )
        df_ = self.circle
        S = sciedm.SMap(columns = 'x', target = 'x',
                        lib = [1,100], pred = [110,160], E = 4, Tp = 1,
                        tau = -1, theta = 3.)
        S.fit(df_)
        pred = S.predict(df_)
        dfv = self.ValidFiles["SMap_circle_E4_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        S2 = round( S.Projection_.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_SMap_2( self ):
        '''SMap embedded = True'''
        if self.verbose : print ( "--- SMap 2 embedded = True ---" )
        df_ = self.circle
        S = sciedm.SMap(columns = ['x', 'y'], target = 'x',
                        lib = [1,200], pred = [1,200], E = 2, Tp = 1,
                        tau = -1, embedded = True, theta = 3.)

        S.fit(df_)
        pred = S.predict(df_)
        dfv  = self.ValidFiles["SMap_circle_E2_embd_valid.csv"]
        
        df  = S.Projection_
        dfc = S.Coefficients_

        S1 = round( dfv.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        S2 = round( df['Predictions'][1:195], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

        self.assertTrue( dfc['∂x/∂x'].mean().round(5) == 0.99801 )
        self.assertTrue( dfc['∂x/∂y'].mean().round(5) == 0.06311 )

    #------------------------------------------------------------
    def test_SMap_3( self ):
        '''SMap nan'''
        if self.verbose : print ( "--- SMap 3 nan ---" )
        df_ = self.circle
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        S = sciedm.SMap(columns = 'x', target = 'y',
                        lib = [1,50], pred = [1,50], E = 2, Tp = 1,
                        tau = -1, theta = 3.)

        S.fit(dfn)
        pred = S.predict(dfn)
        rho = S.score(dfn, dfn['y'])
        dfv = self.ValidFiles["SMap_nan_valid.csv"]
        df  = S.Projection_

        S1 = round( dfv.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        S2 = round( df['Predictions'][1:50], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_SMap_4( self ):
        '''DateTime'''
        if self.verbose : print ( "--- SMap 4 noTime ---" )
        df_ = self.circle_noTime

        S = sciedm.SMap(columns = 'x', target = 'y',
                        lib = [1,100], pred = [101,150], E = 2,
                        theta = 3, noTime = True)

        S.fit(df_)
        pred = S.predict(df_)
        dfv = self.ValidFiles["SMap_noTime_valid.csv"]
        df  = S.Projection_

        S1 = round( dfv.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        S2 = round( df['Predictions'][1:50], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    # CCM
    #------------------------------------------------------------
    def test_CCM_1( self ):
        if self.verbose : print ( "--- CCM 1 ---" )
        df_ = self.sardine_anchovy_sst
        c = sciedm.CCM(columns = 'anchovy',target = 'np_sst',
                       libSizes = [10,20,30,40,50,60,70,75], sample = 100,
                       E = 3, Tp = 0, tau = -1, random_state = 777)
        ccm = c.fit_transform(df_)

        dfv = round( self.ValidFiles["CCM_anch_sst_valid.csv"], 2 )

        self.assertTrue( dfv.equals( round( c.libMeans_, 2 ) ) )

    #------------------------------------------------------------
    def test_CCM_2( self ):
        '''CCM Multivariate'''
        if self.verbose : print ( "--- CCM 2 multivariate ---" )
        df_ = self.Lorenz5D
        c = sciedm.CCM(columns = ['V3','V5'], target = 'V1',
                       libSizes = [20, 200, 500, 950], sample = 30, E = 5,
                       Tp = 10, tau = -5, random_state = 777)
        ccm = c.fit_transform(df_)

        dfv = round( self.ValidFiles["CCM_Lorenz5D_MV_valid.csv"], 4 )

        self.assertTrue( dfv.equals( round( c.libMeans_, 4 ) ) )

    #------------------------------------------------------------
    def test_CCM_3( self ):
        '''CCM nan'''
        if self.verbose : print ( "--- CCM 3 nan ---" )
        df_ = self.circle
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        libSizes = [_ for _ in range(10,191,10)]
        c = sciedm.CCM(columns = 'x', target = 'y',
                       libSizes = libSizes, sample = 20, E = 2,
                       Tp = 5, tau = -1, random_state = 777)
        ccm = c.fit_transform(dfn)

        dfv = round( self.ValidFiles["CCM_nan_valid.csv"], 4 )

        self.assertTrue( dfv.equals( round( c.libMeans_, 4 ) ) )

    #------------------------------------------------------------
    # EmbedDimension
    #------------------------------------------------------------
    def test_EmbedDimension( self ):
        if self.verbose : print ( "--- EmbedDimension ---" )
        df_ = self.Lorenz5D
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = sciedm.EmbedDimension(columns='V1', target='V1',
                                      maxE = 12, lib = [1, 500], pred=[501, 800],
                                      Tp = 15, tau = -5, exclusionRadius = 20,
                                      n_jobs = 10, mpMethod='forkserver')
            edim = e.fit_transform(df_)

        dfv = round( self.ValidFiles["EmbedDim_valid.csv"], 6 )

        self.assertTrue( dfv.equals( round( edim, 6 ) ) )

    #------------------------------------------------------------
    # PredictNonlinear
    #------------------------------------------------------------
    def test_PredictNonlinear( self ):
        if self.verbose : print ( "--- Predict ---" )
        df_ = self.TentMapNoise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = sciedm.PredictNonlinear(columns = 'TentMap', target = 'TentMap',
                                        lib = [1, 500], pred = [501,800], E = 4,
                                        Tp = 1, tau = -1, n_jobs = 10,
                                        theta = [0.01,0.1,0.3,0.5,0.75,1,1.5,
                                                 2,3,4,5,6,7,8,9,10,15,20],
                                        mpMethod='forkserver')
            pnl = p.fit_transform(df_)

        dfv = round( self.ValidFiles["PredictNonlinear_valid.csv"], 6 )

        self.assertTrue( dfv.equals( round( pnl, 6 ) ) )

#------------------------------------------------------------
#
#------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
