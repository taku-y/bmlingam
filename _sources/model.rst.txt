Model
-------------------------------------------------------------------------------
.. contents::
    :depth: 3

Here, equations for the mixed LiNGAM are presented and corresponding command-line options are described. 

In the mixed LiNGAM for two random variables :math:`x_{1}` and :math:`x_{2}`, the relationship of these variables are modeled as:

.. math::

    x_{1}^{(i)} &= \mu_{1} + \tilde{\mu}_{1}^{(i)} + e_{1}^{(i)} \\
    x_{2}^{(i)} &= \mu_{2} + \tilde{\mu}_{2}^{(i)} + 
                   b_{21}x_{1}^{(i)} + e_{2}^{(i)}

or

.. math::

    x_{1}^{(i)} &= \mu_{1} + \tilde{\mu}_{1}^{(i)} + 
                   b_{12}x_{2}^{(i)} + e_{1}^{(i)} \\
    x_{2}^{(i)} &= \mu_{2} + \tilde{\mu}_{2}^{(i)} + e_{2}^{(i)}. 

The former pair of equations, denoted by :math:`{\cal M}_{1}`, models causation from :math:`x_{1}` to :math:`x_{2}`, while the latter one :math:`{\cal M}_{2}` represents the opposite direction. Intercepts :math:`\mu_{1}` and :math:`\mu_{2}`, individual specific effects :math:`\tilde{\mu}_{1}^{(i)}` and :math:`\tilde{\mu}_{2}^{(i)}`, regression coefficients :math:`b_{21}` and :math:`b_{12}` and error variables :math:`e_{1}` and :math:`e_{2}^{(i)}` are unknown variables and should be estimated.  

Causal inference is to choose one of the above models based on observations :math:`{\cal D}=\{(x_{1}^{(i)}, x_{2}^{(i)})\}_{i=1}^{n}`. To do this, we put prior distributions on the unknown variables and performe Bayesian model selection. The marginal likelihood of the above models are:

.. math::

    p({\cal D}|{\cal M}_{1}) &= \int p({\cal D}|\mu, b_{21}, \tilde{\mu}, e,{\cal M}_{1})p(\mu)p(b_{21})p(\tilde{\mu})p(e)d\mu d b_{21} d\tilde{\mu} de \\
    p({\cal D}|{\cal M}_{2}) &= \int p({\cal D}|\mu, b_{12}, \tilde{\mu}, e,{\cal M}_{2})p(\mu)p(b_{12})p(\tilde{\mu})p(e)d\mu d b_{12} d\tilde{\mu} de, 

where :math:`\tilde{\mu}\equiv\{\tilde{\mu}_{1}^{(i)}, \tilde{\mu}_{2}^{(i)}\}_{i=1}^{n}` and :math:`e\equiv\{e_{1}^{(i)}, e_{2}^{(i)}\}_{i=1}^{n}`. In the BMLiNGAM software, the marginal likelihood is calculated by a naive Monte Carlo procedure. 

When a set of prior distributions on all of the unknown variables are fixed, we only need to compare these two quantities. However, appropriate prior distributions are difficult to determine a priori. Thus we apply various sets of prior distributions for :math:`{\cal M}_{1}` and :math:`{\cal M}_{2}` and choose the model with the maximum of the marginal likelihood. 

Specifically, the procedure taken in this software (and the original paper of the mixed LiNGAM) is as follows. For :math:`m` sets of prior distributions, let us denote the two models with the :math:`j`-th prior set (:math:`1\leq j\leq m`) as :math:`{\cal M}_{1,j}` and :math:`{\cal M}_{2,j}`. The optimal models for both causal directions are:

.. math::

    {\cal M}_{1*} &= {\rm arg}\max_{{\cal M}_{1,j}}p({\cal D}|{\cal M}_{1,j}) \\
    {\cal M}_{2*} &= {\rm arg}\max_{{\cal M}_{2,j}}p({\cal D}|{\cal M}_{2,j}). 

Then, :math:`p({\cal D}|{\cal M}_{1*})` and :math:`p({\cal D}|{\cal M}_{2*})` are compared and finally choose the model 

.. math::

    {\cal M}_{*}={\rm arg}\max_{{\cal M}_{1*},{\cal M}_{2*}}(p({\cal D}|{\cal M}_{1*}), p({\cal D}|{\cal M}_{2*})). 

.. note::

    Obviously, it holds that :math:`{\cal M}_{*}={\rm arg}\max_{{\cal M}_{c,j} (c\in\{1,2\}, j=1,\cdots,m)}p({\cal D}|{\cal M}_{c,j})`. 

When assuming parametrized family of prior distributions, as also in this software, this procedure is referred to as hyperparameter optimization. 

.. _posterior-prob-model:

Posterior probability of models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The posterior probability of a model given the data is calculated by normalizing the full probability:

.. math::

    P({\cal M}_{c,j}|{\cal D})=
    \frac{P({\cal D}, {\cal M}_{c,j})}{P({\cal D})}=
    \frac{P({\cal D}|{\cal M}_{c,j})P({\cal M}_{c,j})}{P({\cal D})}, 

where :math:`P({\cal D})=\sum_{c,j}P({\cal D}, {\cal M}_{c,j})`. Since we assume a uniform distribution on :math:`P({\cal M}_{c,j})`, the chance level of the probability with random selection of a model is the reciprocal of the number of the candidate models. 

Standardization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before describing prior distributions, we should mention about standardization of the data. Given :code:`--standardize_on` to :code:`bmlingam-causality`, observations are standardized such that mean 0 and std 1. Although this preprocessing has not been employed in the paper, it is expected that, after standardization, causal inference does not depend on the scale of observed variables. 

Prior distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the following subsections, we describe prior distributions and options of :code:`bmlingam-causality`, which defines possible hyperparameter values. 

Scale parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are two types of scale parameters. The one is common scale parameter :math:`\tau^{cmmn}_{1,2}`, which controls the variance of prior distributions on intercepts, regression coefficients, and error variables. We set :math:`\tau^{cmmn}_{1}=max_c\times\hat{\rm var}(x_{1})` and :math:`\tau^{cmmn}_{2}=max_c\times\hat{\rm var}(x_{2})`, where :math:`\hat{\rm var}(x)` denotes sample variance. :math:`max_{c}` is specified with the option :code:`max_c`. 

The other scale parameters, denoted by :math:`\tau^{indvdl}_{1}` and :math:`\tau^{indvdl}_{2}`, are used for the prior distribution on individual specific effects (described later). We set :math:`\tau^{indvdl}_{1}=c\times\hat{\rm var}(x_{1})` and :math:`\tau^{indvdl}_{2}=c\times\hat{\rm var}(x_{2})`, where :math:`c` is a hyperparameter specified with the option :code:`cs`. This option is a list of positive real values. Each of values in :code:`cs` will be tested. Default is :code:`0, .2, .4, .6, .8`. 

Intercepts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If :code:`--fix_mu_zero_on` is given to :code:`bmlingam-causality`, :math:`\mu_{1}=\mu_{2}=0` (constant). This option is reasonable when standardization is applied. Otherwise (:code:`--fix_mu_zero_off`), normal distributions are assumed as prior on :math:`\mu_{1,2}`:

.. math::

    \mu_{1} &\sim N(0, \tau^{cmmn}_{1}) \\
    \mu_{2} &\sim N(0, \tau^{cmmn}_{2}).

Regression coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Prior on the regression coefficient :math:`b_{21}` or :math:`b_{12}` follows normal distribution:

.. math::

    b_{21} &\sim N(0, \tau^{cmmn}_{2}) \\
    b_{12} &\sim N(0, \tau^{cmmn}_{1}). 

Scale of error varaibles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If :code:`prior_scale=tr_normal`, priors on the scale of error variables are:

.. math::

    \tilde{h_{1}} &\sim N(0,\tau^{cmmn}_{1}) \\
    \tilde{h_{2}} &\sim N(0,\tau^{cmmn}_{2}) \\
    h_{1} &= |\tilde{h}_{1}| \\
    h_{2} &= |\tilde{h}_{2}|. 

If :code:`prior_scale=log_normal`, 

.. math::

    \log h_{1} &\sim N(0,\tau^{cmmn}_{1}) \\
    \log h_{2} &\sim N(0,\tau^{cmmn}_{2}). 

Error variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If :code:`dist_noise=laplace`, 

.. math::

    p(e_{1}) &= Laplace(0, h_{1}/\sqrt{2}) \\
    p(e_{2}) &= Laplace(0, h_{2}/\sqrt{2}) \\

If :code:`dist_noise=gg`, 

.. math::

    p(e_{1}) &= GG(1, m_{1}^{err}, \beta^{err}) \\
    p(e_{2}) &= GG(1, m_{2}^{err}, \beta^{err}), 

where :math:`GG` denotes generalized Gaussian distribution (see below). Possible values of the shape parameter :math:`\beta^{err}` are specified with option :code:`betas_noise`. Default is :code:`.25,.5,.75,1.` (comma-separated float values). The scale parameter :math:`m_{1}^{err}` (:math:`m_{2}^{err}`) is determined such that the variance of :math:`e_{1}` (:math:`e_{2}`) are :math:`h_{1}` (:math:`h_{1}`). 

Individual specific effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Individual specific effects implicitly model correlation of observed variables. To do this, a correlation matrix :math:`L` is introduced, where :math:`L_{pp}=1` (:math:`p=1,2`) and :math:`L_{12}=L_{21}=\sigma_{12}`. :math:`\sigma_{12}` determine the strength of correlation. For hyperparameter optimization, :math:`\sigma_{21}` is varied as :math:`-0.9,-0.7,-0.5,-0.3,0,.3,.5,.7,.9`, which is specified by :code:`--L_cov_21s`. 

The prior distribution on :math:`[\tilde{\mu}_{1}^{(i)}, \tilde{\mu}_{2}^{(i)}]` are chosen from the followings:

- T distribution (default, :code:`--prior_indvdls=t`):

    .. math::

        \left[
            \tilde{\mu}_{1}^{(i)}/\sqrt{\tau_{1}^{indvdl}}, 
            \tilde{\mu}_{2}^{(i)}/\sqrt{\tau_{2}^{indvdl}}
        \right] &\sim T_{\nu}(0, L_{t(\nu)}), 

    where :math:`\nu` is the degrees of freedom of the distribution. Default is :math:`\nu=8`. :math:`L_{t(\nu)}` is proportional to :math:`L` and scaled such that :math:`{\rm var}(\tilde{\mu}_{1, 2}^{(i)}/\sqrt{\tau_{1, 2}^{indvdl}})=1`. Thus, the standard deviation of :math:`\tilde{\mu}_{1}^{(i)}` (:math:`\tilde{\mu}_{2}^{(i)}`) is :math:`\sqrt{\tau}_{1}^{indvdl}` (:math:`\sqrt{\tau}_{2}^{indvdl}`). 

- Normal distribution (:code:`--prior_indvdls=gauss`):

    .. math::

        \left[
            \tilde{\mu}_{1}^{(i)}/\sqrt{\tau_{1}^{indvdl}}, 
            \tilde{\mu}_{2}^{(i)}/\sqrt{\tau_{2}^{indvdl}}
        \right] &\sim N(0, L). 

- Generalized Gaussian distribution (:code:`--prior_indvdls=gg`):
    
    .. math::

        \left[
            \tilde{\mu}_{1}^{(i)}/\sqrt{\tau_{1}^{indvdl}}, 
            \tilde{\mu}_{2}^{(i)}/\sqrt{\tau_{2}^{indvdl}}
        \right] &\sim GG(L, m^{indvdl}, \beta^{indvdl}), 

    where :math:`m^{indvdl}` is determined such that :math:`{\rm var}(\tilde{\mu}_{1, 2}^{(i)}/\sqrt{\tau_{1, 2}^{indvdl}})=1` to variance 1. :math:`\beta^{indvdl}` varies during hyperparameter optimization and possible values are set with :code:`betas_coeff` (default to :code:`.25,.5,.75,1.`). 

Generalized Gaussian distribution 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The density function of :math:`p`-dimensional Generalized Gaussian distribution with mean 0 is defined as [FLJY2013]_:

.. math::

    GG(M,m,\beta) &= p(x|M,m,\beta) \\
                  &= \frac{1}{|M|^{1/2}}h_{m,\beta}(x'M^{-1}x) \\
    h_{m,\beta}(y) &=
        \frac{\beta\Gamma(n/2)}{\pi^{n/2}\Gamma(n/(2\beta))2^{n/(2\beta)}}
        \frac{1}{m^{n/2}}
        \exp\left(-\frac{y^{\beta}}{2m^{\beta}}\right), 

where :math:`M` is the normalized scaling matrix such that :math:`{\rm diag}(M)=p`, :math:`m` is the scale parameter, and :math:`\beta` is the shape parameter. 

.. [FLJY2013] Frédéric Pascal, Lionel Bombrun, Jean-Yves Tourneret and Yannick Berthoumieu. Parameter Estimation For Multivariate Generalized Gaussian Distributions. IEEE Transactions on Signal Processing 23, 2013. 
