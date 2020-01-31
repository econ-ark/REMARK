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
# # Cocco, Gomes, & Maenhout (2005)

# %% [markdown]
# # "[Consumption and Portfolio Choice Over the Life Cycle](https://academic.oup.com/rfs/article-abstract/18/2/491/1599892)"
#
# - Notebook created by Mateo Velásquez-Giraldo and Matthew Zahn.
# - All figures and tables were taken from the linked paper.

# %% [markdown]
# ### Summary
#
# This article uses a dynamic structural model to analyze the optimal portfolio allocation between a risky and a risk-fee asset over the life cycle. It finds that human wealth acts as an imperfect substitute for the risk-free asset, generating an optimal risky asset share that decreases over the life cycle. The model is further used to show that portfolio choices arising from models without life-cycle considerations (e.g., Merton's) generate substantial welfare losses.
#
# The main findings are summarized below.
# - The shape of the income profile over the life span induces investors to reduce proportional stock holdings when aging, which rationalizes the advice from the popular finance literature. 
# - Labor income that is uncorrelated with equity returns is perceived as a closer substitute for risk free asset holdings than equities. Thus, the presence of labor income increases the demand for stocks, particularly earlier in an investor's life. 
# - Even small risks to employment income have a significant effect on investing behavior. Author's describe it has a "crowding" out effect that is particularly strong for younger households. 
# - The lower bound of the income distribution is very relevant to understanding borrowing capacity and portfolio allocations.

# %% [markdown]
# ### The base model
#
# The authors' aim is to represent the life cycle of a consumer that is exposed to uninsurable labor income risk and how he chooses to allocate his savings between a risky and a safe asset, without the possibility to borrow or short-sell.
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
# where $C_{i,t}$ is consumption, $\alpha_{i,t}$ is the share of savings allocated to the risky asset, $Y_{i,t}$ is labor income, and $X_{i,t}$ represents wealth. The utility function $U(\cdot)$ is assumed to be CRRA in the base model. Extensions beyond the baseline model include an additively separable bequest motive in the utility function. The discount factor is $\delta$ and $p_t$ is the probability of survival from $t$ to $t+1$. Death is certain at a maximum period $T$.
#
# Note that the consumer cannot borrow or short-sell.
#
# The control variables in the problem are $\{C_{it}, \alpha_{it}\}^T_{t=1}$ and the state variables are $\{t, X_{it}, v_{it} \}^T_{t=1}$. The agent solves for  policy rules as a function of the state variables&mdash;$C_{it}(X_{it}, v_{it})$ and $\alpha_{it}(X_{it}, v_{it})$. 
#
# #### Labor income
#
# An important driver of the paper's results is the labor income process. It is specified as follows:
#
# \begin{equation}
# \log Y_{i,t} = f(t,Z_{i,t}) + v_{i,t} + \epsilon_{i,t}, \quad \text{for } t\leq K.
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
#
# The borrowing and short-selling constraints ensure that agents cannot allocate negative dollars to either of these assets or borrow against future labor income or retirement wealth. Recall $\alpha_{i,t}$ is the proportion of the investor's savings that are invested in the risky asset. The model's constraints imply that $\alpha_{i,t}\in[0,1]$ and wealth is non-negative.
#

# %% [markdown]
# ### Calibration
#
# __Labor income process__
#
# The PSID is used to estimate the labor income equation and its permanent component. This estimation controls for family specific fixed effects. In order to control for education, the sample was split into three groups: no high school, high school but no college degree, and college graduates. Across each of these groups, $f(t,Z_{i,t})$ is assumed to be additively separable across its arguments. The vector of personal characteristics $Z_{i,t}$ includes age, household fixed effects, marital status, household size/composition. The sample uses households that have a head between the age of 20 and 65. For the retirement stage, $\lambda$ is calibrated as the ratio of the average of labor income in a given education group to the average labor income in the last year of work before retirement. 
#
# The error structure of the labor income process is estimated by following the variance decomposition method described in Carroll and Samwick (1997). A similar method is used to estimate the correlation parameter $\rho$. Define $r_{i,d}$ as:
#
# \begin{eqnarray*}
# r_{id} \equiv \log(Y^*_{i,t+d}) - \log(Y^*_{i,t}), \text{ }d\in \{1,2,...,22\}. \\
# \end{eqnarray*}
#
# Where $Y^*_t$,
# \begin{eqnarray*}
# \log(Y^*_{i,t}) \equiv \log(Y_{i,t}) - f(t,Z_{i,t}).
# \end{eqnarray*}
# Then,
# \begin{eqnarray*}
# \text{VAR}(R_{i,d}) = d*\sigma^2_u + 2*\sigma^2_\epsilon.
# \end{eqnarray*}
#
# The variance estimates can be obtained via an OLS regression of $\text{VAR}(R_{i,d})$ on $d$ and a constant term. These estimated values are assumed to be the same across all individuals. For the correlation parameter, start by writing the change in $\log(Y_{i,t})$ as:
#
# \begin{eqnarray*}
# r_{i,1} = \xi_t + \omega_{i,t} + \epsilon_{i,t} - \epsilon_{i,t-1}
# \end{eqnarray*}
#
# Averaging across individuals gives:
#
# \begin{eqnarray*}
# \bar{r_1} = \xi_t
# \end{eqnarray*}
#
# The correlation coefficient is also obtained via OLS by regressing $\overline{\Delta \log(Y^*_t)}$ on demeaned excess returns:
#
# \begin{eqnarray*}
# \bar{r_1} = \beta(R_{t+1} - \bf{\bar{R}}_f - \mu) + \psi_t
# \end{eqnarray*}
#
# __Other parameters__
#
# Adults start at age 20 without a college degree and age 22 with a college degree. The retirement age is 65 for all households. The investor will die for sure if they reach age 100. Prior to this age, survival probabilities come from the mortality tables published by the National Center for Health Statistics. The discount factor $\delta$ is calibrated to be $0.96$ and the coefficient of relative risk aversion ($\gamma$) is set to $10$. The mean equity premium $\mu$ is $4%$, the risk free rate is $2%$, and the standard deviation of innovations to the risky asset is set to the historical value of $0.157$.
#
# For reference, the authors' source Fortran code that includes these paramerization details is available on [Gomes' personal page](http://faculty.london.edu/fgomes/research.html). Code that solves the model is also available in [Julia](https://github.com/econ-ark/HARK/issues/114#issuecomment-371891418).

# %% [markdown]
# ### Key Results
#
# #### The optimal risky asset share
#
# The figure below shows the policy function for the risky portfolio share as a function of wealth at different ages.
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
#
# #### Heterogeneity and sensitivity analysis
#
# The authors also considered a number of extensions to the baseline model. These are summaried below along with their main conclusions. 
#
# - Labor income risk: Income risk may vary across employment sectors relative to the baseline model. The authors examine extreme cases for industries that have a large standard deviation and temporary income shocks. While some differences appear across sectors, the results are generally in line with the baseline model.
# - Disastrous labor income shocks: The authors find that even a small probability of zero labor income lowers the optimal portfolio allocation in stocks, while the qualitative features of the baseline model are preserved.
# - Uncertain retirement income: The authors consider two types of uncertainty for retirement income; it is stochastic and correlated with current stock market performance and allowing for disastrous labor income draws before retirement. The first extension has results essentially the same as the baseline case. The second leads to more conservative portfolio allocations but is broadly consistent with the baseline model.
# - Endogenous borrowing constraints: The authors add borrowing to their model by building on credit-market imperfections. They find that the average investor borrows about \$5,000 and are in debt for most of their working life. The agents eventually pay off this debt and save for retirement. Relative to the benchmark model, the investor has put less of their money in their portfolio and arrive at retirement with substantially less wealth. These results are particularly pronounced at the lower end of the income distribution relative to the higher end. Additional details are available in the text.
# - Bequest motive: The authors introduce a bequest motive into the agent's utility function (i.e., $b>0$). Young investors are more impatient and tend to save less for bequests. As the agent ages, savings increases and is strongest once the agent retires. This leads to effects on the agent's portfolio allocation. Taking a step-back however, these effects are not very large unless $b$ is large. 
# - Educational attainment: The authors generally find that savings are consistent across education groups. They note that for a given age, the importance of future income is increasing with education level. This implies that riskless asset holdings are larger for these households. 
# - Risk aversion and intertemporal substitution: Lowering the level of risk aversion in the model leads to changes in the optimal portfolio allocation and wealth accumulation. Less risk-averse investors accumulate less precautionary savings and invest more in risky assets.
#

# %% [markdown]
# ### Conclusion
#
# This article provides a dynamic model with accurate lifetime income profiles in which labor income increases risky asset holdings, as it is seen as a closer substitute of risk-free assets. It finds an optimal risky asset share that decreases in wealth and with age, after middle age. The model is also used to show that ignoring labor income for portfolio allocation can generate substantial welfare losses.
