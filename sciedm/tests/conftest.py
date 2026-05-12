"""
conftest.py — shared resources for sciedm pytest suite.

Contents:
  - GetMP_ContextName()  multiprocessing context helper
  - ValidData()          load a validation CSV by filename
  - *Args dicts          default keyword arguments for each EDM API function
"""

import os
from multiprocessing import get_context, get_start_method

from pandas import read_csv
import sciedm

# ---------------------------------------------------------------------------
# Multiprocessing context helper  (remove when > Python 3.13)
# ---------------------------------------------------------------------------

def GetMP_ContextName():
    '''Until > Python 3.14, disallow "fork" multiprocessing context.'''
    allowedContext = ("forkserver", "spawn")
    current = get_start_method( allow_none = True )
    if current in allowedContext:
        return get_context( current )._name
    for method in allowedContext:
        try:
            return get_context( method )._name
        except ValueError:
            continue

# ---------------------------------------------------------------------------
# Validation file helper
# ---------------------------------------------------------------------------

VALID_DIR = os.path.join( os.path.dirname( sciedm.__file__ ),
                          "tests", "validation" )

def ValidData( filename ):
    '''Return validation CSV DataFrame from sciedm validation/ directory.'''
    return read_csv( os.path.join( VALID_DIR, filename ) )

# ---------------------------------------------------------------------------
# Default argument dictionaries — one per API function.
#
# Every parameter is listed. Parameters not actively tested carry a comment.
# Tests copy the relevant dict and update only the parameters under test,
# making each test's variation immediately visible.
# ---------------------------------------------------------------------------

SimplexArgs = dict( columns         = None,
                    target          = None,
                    E               = 1,
                    tau             = -1,
                    Tp              = 1,
                    lib             = None,
                    pred            = None,
                    knn             = 0,
                    exclusionRadius = 0,
                    embedded        = False,
                    noTime          = False )

SMapArgs = dict( columns         = None,
                 target          = None,
                 E               = 1,
                 tau             = -1,
                 Tp              = 1,
                 lib             = None,
                 pred            = None,
                 theta           = 0.0,
                 solver          = None,
                 knn             = 0,
                 exclusionRadius = 0,
                 embedded        = False,
                 noTime          = False )

CCMArgs = dict( columns         = None,
                target          = None,
                E               = 1,
                libSizes        = None,
                Tp              = 0,
                tau             = -1,
                knn             = 0,
                sample          = 30,
                random_state    = None,
                exclusionRadius = 0,
                validLib        = [],
                embedded        = False,
                noTime          = False,
                includeData     = False,
                mpMethod        = GetMP_ContextName(),
                sharedMB        = 0.01,
                parallel        = False,
                verbose         = False )

CCM_MatrixArgs = dict( E                = None,
                       libSizes         = [],
                       pLibSizes        = [10,20,80,90],
                       Tp               = 0,
                       tau              = -1,
                       exclusionRadius  = 0,
                       sample           = 30,
                       seed             = None,
                       noTime           = False,
                       parallel         = True,
                       mpMethod         = GetMP_ContextName(),
                       sharedMB         = 0.01,
                       targetBatchSize  = None,
                       expConverge      = False,
                       progressLog      = None,
                       progressInterval = 10 )

EmbedDimensionArgs = dict( columns         = None,
                           target          = None,
                           maxE            = 10,
                           lib             = None,
                           pred            = None,
                           Tp              = 1,
                           tau             = -1,
                           exclusionRadius = 0,
                           embedded        = False,
                           noTime          = False,
                           mpMethod        = GetMP_ContextName(),
                           chunksize       = 1,
                           n_jobs          = 10 )

PredictNonlinearArgs = dict( columns         = None,
                             target          = None,
                             theta           = [0.01, 0.1, 0.3, 0.5, 0.75, 1,
                                                1.5, 2, 3, 4, 5, 6, 7, 8, 9],
                             E               = 1,
                             lib             = None,
                             pred            = None,
                             Tp              = 1,
                             tau             = -1,
                             exclusionRadius = 0,
                             embedded        = False,
                             noTime          = False,
                             mpMethod        = GetMP_ContextName(),
                             chunksize       = 1,
                             n_jobs          = 10 )
