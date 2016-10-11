Accuracy of BMLiNGAM on artificial dataset
-------------------------------------------------------------------------------
We have tested estimation accuracy of BMLiNGAM on artificial datasets with 100 observations. The accuracy of the causal inference  (:math:`x_{1}\rightarrow x_{2}` or vice versa) was over 80\%. The datasets were created in the same manner as (Shimizu and Bollen, 2014). 

To qualitatively check the accuracy of the estimation of posterior distribution of the regression coefficient, we plotted posterior means of the coefficients versus the true value. As show in the below figure, the sign of the regression coefficient was reasonably estimated, although its amplitude was not very accurate. 

.. image:: postmeans.png
    :align: center
    :width: 50%

The following HTML files show results of the simulations to check the estimation accuracy:

- :download:`Accuracy of causality estimation <accuracy_causality_estimation.html>`
- :download:`Accuracy of estimation of regression coefficient as posterior mean (0 confounder) <post_coeff_conf0.html>`
- :download:`Accuracy of estimation of regression coefficient as posterior mean (6 confounders) <post_coeff_conf6.html>`
