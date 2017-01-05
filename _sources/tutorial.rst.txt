Tutorial
-------------------------------------------------------------------------------
You can download history of the commands used in the following tutorial: :download:`commands.txt`. In the tutorial, you will create an artificial data :code:`sampledata.csv`. If you want to analyze your data, replace :code:`sampledata.csv` by the file name of your data in the arguments of the commands. 

Data format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BMLiNGAM accepts CSV files consisting of sample values, where each column corresponds to a variable. At least two columns should be contained in the given CSV file. If the file has three or more columns, BMLiNGAM analyzes all pairs in the file. The first row of the file is regarded as a header representing names of the variables. 

For testing purposes, you can create a sample data file:

.. code-block:: console

    $ bmlingam-make-testdata

This command creates :code:`sampledata.csv` in the current directory. The first few lines of the file are look like:

.. literalinclude:: commands.txt
    :language: console
    :start-after: # head -6 sampledata.csv
    :end-before: # bmlingam-causality sampledata.csv --result_dir=result1

'x1_src' and 'x2_dst' are the names of the two variables. Since this data is artificially created, the source (cause) and destination (effect) of the causal relationship is known. To check the estimation result, '(src)' and '(dst)' are included in the variable names. 

When analyzing your own data, you should make your data as shown above. 

Causal inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To estimate causality for the data: 

.. code-block:: console

    $ bmlingam-causality sampledata.csv --result_dir=result1

You will get a message as follows:

.. literalinclude:: commands.txt
    :language: console
    :start-after: # bmlingam-causality sampledata.csv --result_dir=result1
    :end-before: # bmlingam-causality --help

This message means that the estimated causal direction is x1_src -> x2_dst with the posterior probability  (:math:`P({\cal M}_{*}|{\cal D})`, see :ref:`posterior-prob-model` for its definition) :posteriorprob:`dummy`. We will describe the interpretation of this probability later. You can also see the marginal log likelihood :loglikelihood:`dummy`.

The message also includes the posterior probability of the optimal reverse model, which has the highest posterior probability in the models with the reverse causal direction to the optimal model :math:`{\cal M}_{*}`. In the above example, the posterior probability the optimal reverse model is almost 0 (see :code:`causality.csv` to know the accurate value, later). 

After posterior probabilities, hyper parameters of these (global and reverse) optimal models are shown. Here is a table describing about hyper parameters:  

- :code:`prior_indvdl`: Distribution of individual specific effects. :code:`'t'` or :code:`'gauss'`. 
- :code:`v_indvdl_1/2`: Scale of individual specific effects. These values are used in the code as: 

    .. code-block:: python

            # in bmlingam_np.py
            v_indvdl_1 = (std_x[0] * hparams['v_indvdl_1'])**2
            v_indvdl_2 = (std_x[1] * hparams['v_indvdl_2'])**2
            mu1_ = np.sqrt(v_indvdl_1) * L_mu1
            mu2_ = np.sqrt(v_indvdl_2) * L_mu2

    :code:`mu1_` and :code:`mu2_` correspond to the means of :math:`\tilde{\mu}_{1}^{(i)}` and :math:`\tilde{\mu}_{2}^{(i)}`, respectively, in the paper. 
- :code:`L_cov12/21`: :math:`\sigma_{12}` in the paper. 

If the given data file consists of three columns, all of combinations of the variables will be tested, thus the message will look like as below:

.. literalinclude:: commands.txt
    :language: console
    :start-after: # bmlingam-causality sampledata_threecolumns.csv --n_mc_samples=1000 --result_dir=result_threecol

Command :code:`bmlingam-causality` creates two types of files. One is :code:`causality.csv`, consisting of the estimated causality (direction) for every pairs of the variables. Another is referred to as **model selection file**  with extension '.bmlm.pklz'. A model selection file contains the result of Bayesian model selection, which was applied for causal inference. This file will be used later in estimation of the posterior distribution of the regression coefficient, which is relating the two variables. 

Model selection files will be created for every pairs of the all variables in the data. In the above first example, single file :code:`x1_src_x2_dst.bmlm.pklz` will be created, because :code:`sampledata.csv` has only two variables (two columns). The filename of each model selection file includes the names of the variables in each pair. 

Command line options of :code:`bmlingam-causality` are as follows:

.. literalinclude:: commands.txt
    :language: console
    :start-after: # bmlingam-causality --help
    :end-before: # bmlingam-coeff result1/x1_src_x2_dst.bmlm.pklz

For example, if you want to estimate causality for models Gaussian individual effects:

.. code-block:: console

     bmlingam-causality --prior_indvdls=gauss sampledata.csv

Many of command line parameters are related to model selection procedure. See :doc:`model` in detail. 

Bayesian model selection for causal inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
BMLiNGAM estimates causality with Bayesian model selection (Shimizu and Bollen, 2014). Each Bayesian hierarchical model in candidates (denoted by :math:`\{{\cal M}_{r}\}_{r=1}^{R}`) for the model selection represents causality between two variables and its direction: :math:`x_{1}\rightarrow x_{2}` or :math:`x_{2}\rightarrow x_{1}`. The candidate models also differ in their hyperparameters, which specifies properties other than the causality: covariance of noise, confounders, and so on. 

The causality is inferred with the MAP model, i.e., :math:`r^{*}\equiv{\rm arg}\max_{r}P({\cal M}_{r}|{\cal D})\propto P({\cal M}_{r}, {\cal D})`, where :math:`{\cal D}` denotes the data. If :math:`{\cal M}_{r^{*}}` represents the causality :math:`x_{1}\rightarrow x_{2}`, it is the inference result of BMLiNGAM. 

The posterior probability of a model given the data is calculated by normalizing the full probability:

.. math::

    P({\cal M}_{r}|{\cal D})=
    \frac{P({\cal D}, {\cal M}_{r})}{P({\cal D})}=
    \frac{P({\cal D}|{\cal M}_{r})P({\cal M}_{r})}{P({\cal D})}, 

where :math:`P({\cal D})=\sum_{r}P({\cal D}, {\cal M}_{r})`. Since we assume a uniform distribution on :math:`P({\cal M}_{r})`, the chance level of the probability with random selection of a model is the reciprocal of the number of the candidate models. In the above example, it is 1/450 :math:`\sim` 0.002, which is much smaller than the estimated posterior probability :posteriorprob:`dummy`. 

Posterior distribution of regression coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The posterior distribution of the regression coefficient can be estimated for the selected model: 

.. code-block:: console

    bmlingam-coeff x1_src_x2_dst.bmlm.pklz

A kernel density estimator is applied to the samples from the posterior distribution obtained with MCMC (Metropolis-Hastings algorithm). You will get messages as follows:

.. literalinclude:: commands.txt
    :language: console
    :start-after: # bmlingam-coeff result1/x1_src_x2_dst.bmlm.pklz
    :end-before: # bmlingam-coeff --help

Following indicators shoing the progress of MCMC (including burn-in period), you can see the mean value and 95% credible interval of the MCMC posterior distribution of the regression coefficient. 

This command will also show a figure of the estimated posterior distribution like this: 

.. image:: example-posterior.png
    :align: center
    :width: 50%

Dotted lines indicate 95% credible interval. The MCMC samples is saved in a CSV file with suffix :code:`.bmlm.post.csv`. 

Command line options of :code:`bmlingam-coeff` are as follows:

.. literalinclude:: commands.txt
    :language: console
    :start-after: # bmlingam-coeff --help
    :end-before: # bmlingam-causality sampledata_threecolumns.csv --n_mc_samples=1000 --result_dir=result_threecol

If you want a png image:

.. code-block:: console

    bmlingam-coeff --figtype='png' x1_src_x2_dst.bmlm.pklz
