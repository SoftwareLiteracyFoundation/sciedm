.. _general_examples:

Introductory examples
=====================

Trivial examples based on two data sets, a 5-dimensional Lorenz'96 model and
hydraulic flow into Everglades National Park through three flow control spillways.
An introduction to Empirical Dynamic Modeling (EDM) can be found in the
`EDM docs <https://sugiharalab.github.io/EDM_Documentation/>`_ and general information
in `Wikipedia <https://en.wikipedia.org/wiki/Empirical_dynamic_modeling>`_.

1. `ccm_example.py`
   Compute convergent cross mapping (CCM) between two components of the Lorenz'5D
2. `embed_dimension_example.py`
   Estimate embedding dimension (`E`) of the Everglades flow data
3. `predict_nonlinear_example.py`
   Examine nonlinearity through the `S-Map` `theta` parameter
4. `simplex_example.py`
   Four examples using `Simplex` for out of sample prediction of Lorenz'96 variables.
   
   a. Prediction of V3 from time delay embedding of V3
   b. Prediction of V3 from time delay embedding of V1 (cross mapping)
   c. Prediction of V3 from multivariate embedding of V1,V2,V4,V5
   d. Prediction of V3 from mixed-multivariate embedding of V1,V2,V4,V5
      
5. `smap_example.py`
   Two examples using `SMap` for out of sample prediction of Lorenz'96 variables.
   
   a. Prediction and Jacobians of V3 from time delay embedding of V3
   b. Prediction and Jacobians of V3 from multivariate embedding of V1,V2,V4,V5
