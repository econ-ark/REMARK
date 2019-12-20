# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cocco, Gomes & Maenhout

# %% [markdown]
# # "[Consumption and Portfolio Choice over the Life Cycle](https://academic.oup.com/rfs/article-abstract/18/2/491/1599892)"
#
# - Notebook created by Mateo Velásquez-Giraldo
# - All figures and tables were taken from the linked paper.

# %% [markdown]
# ### Summary
#
# This article uses dynamic structural model to analyze the optimal portfolio allocation between a risky and a risk-fee asset over the life cycle. It finds that human wealth acts as an imperfect substitute for the risk-free asset, generating an optimal risky asset share that decreases over the life cycle. The model is further used to show that portfolio choices arising from models without life-cycle considerations (e.g, Merton's) generate substantial welfare losses.
#

# %% [markdown]
# ### The base model
#
# The authors' aim is to represent the life cycle of a consumer that is exposed to uninsurable labor income risk and chooses how to allocate his savings between a risky and a safe asset, without the possibility to borrow or short sell.
#
# ##### Dynamic problem
#
# The problem of an agent $i$ of age $t$ in the base model is recursively represented as
#
# \begin{split}
# V_{i,t} =& \max_{0\leq C_{i,t} \leq X_{i,t}, \alpha_{i,t}\in[0,1]} U(C_{i,t}) + \delta p_t E_t\{ V_{i,t+1} (X_{i,t+1}) \}\\
# &\text{s.t}\\
# &X_{i,t+1} = Y_{i,t+1} + (X_{i,t} - C_{i,t})(\alpha_{i,t} R_{t+1} + (1-\alpha_{i,t})\bar{R}_f)
# \end{split}
#
# where $C_{i,t}$ is consumption, $\alpha_{i,t}$ is the share of savings allocated to the risky asset, $Y_{i,t}$ is labor income, and $X_{i,t}$ represents wealth. The utility function $U(\cdot)$ is assumed to be CRRA in the base model. The discount factor is $\delta$ and $p_t$ is the probability of survival from $t$ to $t+1$. Death is certain at a maximum period $T$.
#
# Note that the consumer can not borrow or short-sell.
#
# #### Labor income
#
# An important driver of the paper's results is the labor income process. It is specified as follows:
#
# \begin{equation}
# \ln Y_{i,t} = f(t,Z_{i,t}) + v_{i,t} + \epsilon_{i,t}, \quad \text{for } t\leq K.
# \end{equation}
#
# where $K$ is the (exogenous) age of retirement, $Z_{i,t}$ are demographic characteristics, $\epsilon_{i,t}\sim \mathcal{N}(0,\sigma^2_\epsilon)$ is a transitory shock, and  $v_{i,t}$ is a permanent component following a random walk
#
# \begin{equation}
# v_{i,t} = v_{i,t-1} + u_{i,t} = v_{i,t-1} + \xi_t + \omega_{i,t}
# \end{equation}
#
# in which the innovation is decomposed into an aggregate ($\xi_t$) and an idiosyncratic component ($\omega_{i,t}$), both following mean-0 normal distributions.
#
# Post-retirement income is a constant fraction $\lambda$ of income in the last working year $K$.
#
# A crucial aspect of the labor income process is that $f(\cdot,\cdot)$ is calibrated to match income profiles in the PSID, capturing the usual humped shape of income across lifetime.
#
# #### Assets and their returns
#
# There are two assets available for consumers to allocate their savings.
#
# - Bonds: paying a risk-free return $\bar{R}_f$.
#
# - Stocks: paying a stochastic return $R_t = \bar{R}_f + \mu + \eta_t$, where the stochastic component $\eta_t \sim \mathcal{N}(0, \sigma^2_\eta)$ is allowed to be correlated with the aggregate labor income innovation $\xi_t$.

# %% [markdown]
# ### Key Results
#
# I now report the main results of the article.
#
# #### The optimal risky asset share
#
# The next figure shows the policy function for the risky portfolio share as a function of wealth at different ages.
#
# <center><img src="Figures\Opt_shares_by_age.jpg" style="height:250px"></center>
#
# The optimal risky share is decreasing in wealth. The authors argue this is due to the fact that, at low levels of wealth, relatively safe human wealth represents a higher fraction of the consumer's wealth, so he shifts his tradeable wealth towards riskier alternatives.
#
# Analyzing the policy rule by age also shows that the risky share increases from young to middle age, and decreases from middle to old age. This is consistent with the previous interpretation: shares trace the humped shape of labor earnings.
#
# #### The welfare implications of different allocation rules
#
# The authors next conduct a welfare analysis of different allocation rules, including popular heuristics. The rules are presented in the next figure.
#
# <center><img src="Figures\Alloc_rules.jpg" style="height:500px"></center>
#
# The utility cost of each policy in terms of constant consumption streams with respect to the authors calculated optimal policy function is reported in the next table.
#
# <center><img src="Figures\Util_cost.jpg" style="height:100px"></center>
#
# Interestingly, the "no-income" column corresponds to the usual portfolio choice result of the optimal share being the quotient of excess returns and risk times relative risk aversion, disregarding labor income. The experiment shows this allocation produces substantial welfare losses.

# %% [markdown]
# ### Conclusion
#
# This article provides a dynamic model with accurate lifetime income profiles in which labor income increases risky asset holdings, as it is seen as a closer substitute of risk-free assets. It finds an optimal risky asset share that decreases in wealth and with age, after middle age. The model is also used to show that ignoring labor income for portfolio allocation can generate substantial welfare losses.
