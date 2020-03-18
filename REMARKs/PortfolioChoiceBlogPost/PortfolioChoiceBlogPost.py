# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.6
#   latex_envs:
#     LaTeX_envs_menu_present: true
#     autoclose: false
#     autocomplete: false
#     bibliofile: biblio.bib
#     cite_by: apalike
#     current_citInitial: 1
#     eqLabelWithNumbers: true
#     eqNumInitial: 1
#     hotkeys:
#       equation: Ctrl-E
#       itemize: Ctrl-I
#     labels_anchors: false
#     latex_user_defs: false
#     report_style_numbering: false
#     user_envs_cfg: false
# ---

# %% [markdown]
# # Optimal Financial Investment over the Life Cycle
#
# Economists like to compare actual human behavior to the choices that would be made by a "rational" agent who optimally takes into account all the complexities of a decision.
#
# But for some problems, finding the optimal choice can be remarkably difficult.  
#
# Financial choice over the life cycle is such a problem.  Determining the optimal amount to save for retirement, and how much of those savings to invest in risky assets, turns out to be mathematically **much** harder than performing the calculations required to land the Apollo spacecraft on the moon.  Indeed, the computational  tools that economists use to solve such problems descend directly from those originally developed for optimizing spacecraft trajectories in the 1950s.  
#
# By 2005, after 50 years of development, those tools were finally good enough for their advice to be taken seriously -- if by "taken seriously" we mean that economists's own personal choices (and the advice they give to friends and family) were influenced by the results: An academic paper by Cocco, Gomes, and Maenhout (2005) is now the standard reference.
#
# Even today, these tools are so difficult that it can take years of work for a new researcher to master them.  
#
# <!-- 
# Partly this is because each economist (or team of coauthors) has tended to construct from scratch their own complex body of computer code tailored to solving exactly the problem they are interested in.
# -->
#
# Thanks to funding by the Sloan Foundation and the Think Forward Institute, this blog post introduces a set of open-source (free) tools designed to make solving such problems much easier.  Our hope is that eventually such transparent and publicly available tools will provide an alternative to the proprietary tools that have become widely available from so-called "robo advisors" in the last few years.
#
# <!-- 
# The aim of the [Econ-ARK](https://econ-ark.org) toolkit is to address that problem by providing a set of open-source software tools that can solve many computationally difficult models of optimal consumer choice; one of the early contributions of the toolkit was a tool that calculates the optimal path of retirement saving under fairly realistic assumptions  (see below for details).  One of the reasons these conditions are only "fairly" relaistic, though is that the model makes the assumption that the consumer cannot invest any of those savings "risky" assets that are expected to earn a higher rate of return than, say, a bank account ("safe" assets). 
#
# This blog post introduces a new tool in the toolkit, which extends the life cycle/retirement saving model to incorporates the optimal choice of portfolio share between "risky" and "safe" assets.  The development of this tool was generously funded by the Think Forward Institute.
# -->

# %% [markdown]
# ## The Problem
#
# Nobody saving for retirement knows exactly what the future holds. They are likely to change jobs several times during their career, and each new job will have a different profile of income growth and risk; they might develop health problems that cut short their life even before they reach retirement (in which case retirement savings would  be unnecessary), or they might turn out to be so healthy that they live to 100 (in which case there is a danger of outliving their savings).  
#
# Nor does anybody know what the payoffs will turn out to be for alternative investment choices.  Historically, "risky" assets like stocks have on average earned higher returns than "safe" assets like government bonds -- but there is no guarantee that will be true over any particular period.
#
# Uncertainties of these (and other) kinds are why the consumer's problem is so much harder than NASA's: The motion of a spacecraft is almost perfectly predictable using Newton's equations.  If you set it in motion in a certain direction and with a certain velocity, those equations will predict to within a matter of inches where it will be weeks in the future.
#
# The way economists calculate "optimal" behavior over the lifetime begins with an attempt to quantify each of the risks. For example, from available datasets we can calculate how often people change jobs at each age of life, taking account of personal characteristics like education, occupation and so on, and measure what happens to their income after job changes.  Job-related income uncertainty can thus be represented mathematically as a statistical distribution over the many possible future outcomes, and similarly for other kinds of risk (like health risk). When all the biggest individual risks have been quantified, we can calculate the joint probabilities of every conceivable draw of the risks, and weight each possible outcome by its probability.  Finally, we can calculate how the outcomes (like, retirement income) depend on the choice of saving and portfolio choice, and compute which choices would are the best ones for consumers with different preferences (toward risk, for example) or in different circumstances.
#

# %% [markdown]
# ## The Solution
#
# ### Replicating the Standard Model
# Our first step has been to replicate the results of the above-mentioned 2005 paper (by Cocco, Gomes, and Maenhout - "CGM" for short) using our new tool.  Although our results are different in some minor respects from theirs (likely having to do with different assumptions about the details of the risks), the "big-picture" findings are the same.
#
# A key input to such models is a measure of the degree of consumers' ["risk aversion"](https://en.wikipedia.org/wiki/Risk_aversion).  Roughly speaking, your degree of risk aversion determines how much you are willing to pay to buy insurance against financial risks; or, if insurance is not available, how much you alter your behavior (for example, by saving more) as a precaution against the risk.  Researchers have found that many kinds of consumer behavior are consistent with values of the "relative risk aversion parameter" in the range from 2 to 4.  
#
# The most striking conclusion of the CGM paper is captured in the figure below.  Consumers with risk aversion of 3 can choose between a "risky" asset with expected performance (for risk and return) like the stock market and a "safe" asset with lower expected returns.  The figure shows the optimal risky share -- that is, the optimal proportion of savings to invest in a risky asset -- by age.  The fact that it is stuck at 1.0 at every age means that the optimal choice is always to invest 100 percent of your savings in stocks!
#
# ![RShare_CRRA_3](figures/figure_CRRA_3/RShare_Means.png)

# %% [markdown]
# Of course, the optimal portfolio share to put in the risky asset depends on your perception of the degree of riskiness and the average extra return you expect stocks will yield over the long run (the "equity premium").
#
# The CGM model assumes that in the future the average equity premium and the degree of stock market riskiness will be similar to their values in the past.  Specifically, they assume that people expect an equity premium of 4 percent, because that is a good estimate of what the average premium has been on stock market investments in the developed world over the past century.
#
# The conclusion is that for values of risk aversion that describe people's choices well in other contexts, an equity premium of 4 percent a year is more than enough to compensate any rational agent for bearing the risk associated with that return.

# %% [markdown]
# ## Maybe Risk Aversion is Much Greater than 3?
#
# Parameters like "relative risk aversion" are slippery things to measure and calibrate.  Maybe the conventional choice of 3, which works well in other contexts, is inappropriate here -- maybe people just hate stock market risk much more than they hate other kinds of risk.
#
# The next figure shows the profile of the mean risky share when risk aversion is 6, twice the conventional value.
#
# Now the model says that until about age 35, it is still optimal to invest all of your savings in the stock market.  After that, the risky share declines gradually until it stabilizes at around 65 percent at age 65.  (The dashing lines show the choices made by people at the 5th and 95th percentiles of the distribution of the risky share).
#
# These results reflect two aspects of the model:
# 1. Young people are assumed to start with little or no assets
#    * Their income comes mostly from working in the labor market
#    * If you have only a small amount of wealth, the absolute dollar size of the risk you are taking by investing in the risky asset is small, so the higher returns more than make up for the (small) risk
# 1. By the age of retirement, you plan to finance a lot of your spending from your savings
#    * Investing everything in the stock market (like a young person) would put a large proportion of your retirement spending at risk from the fluctuations in the market
#    * The "equity premium" is nevertheless large enough to make it worthwhile for most people to keep more than half of their assets in stocks
#
# ![Parameters_base](figures/figure_Parameters_base/RShare_Means.png)

# %% [markdown]
# ## What Do People Actually Do?
#
# The pattern above is strikingly different from the actual choices that typical savers make.  
#
# The figure below shows data, from the Federal Reserve's triennial [_Survey of Consumer Finances_](https://en.wikipedia.org/wiki/Survey_of_Consumer_Finances), for the proportion of their assets that people at different ages actually have invested in stocks and other risky assets.
#
# The actual profile of the risky share is much lower than the model says is optimal (even with risk aversion of 6).
#
# Two main interpretations are possible:
# 1. The model is basically the right framework for thinking about these questions
#     * But some of its assumptions/calibrations are wrong
# 1. People _are_ behaving optimally, but the model is still missing some important features of reality
#

# %% [markdown]
# ### What Assumptions Might Be Wrong?
#
#

# %% [markdown]
# #### Maybe People Are Pessimistic About The Future Equity Premium
#
# While 4 percent is a reasonable estimate of the equity premium in the past, it is possible that people do not believe it will be as large in the future.
#
# The figure below shows the consequences if people believe the equity premium will be only two percent -- which is within the range of values that some respected economists think might prevail in the future.
#
# The shape of the figure is much the same as before; in particular, the youngest people still hold 100 percent of their portfolios in risky assets.  But the proportion of their portfolios that middle-aged and older people hold in equities falls from about 50 to about 20 percent.
#
# ![RShare_Means](figures/figure_equity_0p02/RShare_Means.png)
#

# %% [markdown]
# #### Is Pessimism Enough?
#
# The immediately preceding figure uses the baseline assumption that relative risk aversion is very high (6).  A natural question is whether when people are pessimistic about the equity premium, their optimal portfolio shares might be low even at a less extreme degree of risk aversion.  
#
# Nope.  The figure below shows that, even with pessimistic beliefs about the equity premium, if relative risk aversion has a conventional value of 3 then the optimal risky share is still 100 percent for both young and old people, and reaches a low point of about 70 percent for people nearing retirement.
#
# ![CRRA_3_Equity_Premium_1](figures/figure_CRRA_3_Equity_Premium_1/RShare_Means.png)

# %% [markdown]
# ### Comparison to Professional Advice

# %% [markdown]
# Another interesting comparison is to the advice of professional investment advisors.  Though that advice can be quite sophistcated and nuanced, it is also sometimes codified in simple rules of thumb.  One of the most common of these is the "100 minus age" rule, which says that the percentage of your portfolio in risky assets should be equal to 100 minus your age, so that a 25 year old would have 75 percent in stocks while a 60 year old would have 40 percent in stocks.
#
# For people before retirement, at least the shape of the profile that advisors recommend is somewhat like the shape that comes out of the model.  While the rule would say that the 25 year old should put 75 percent of their savings in the stock market and the model says 100 percent, they agree that the proportion should be high, and also agree that the proportion should decline during your working life.
#
# However, the rule and the model disagree about what should happen after retirement.  The rule recommends steadily reducing your exposure to risky assets as you get older, while the model says that your exposure should remain at about the same level it was at late in your working life.
#
# The advisors, who have daily contact with real human beings, probably have an insight that the model does not incorporate:  Risk aversion may increase as you get older.  
#
# Risk aversion is remarkably difficult to measure, and economists' efforts to determine whether it increases with age have been inconclusive, with some papers finding [evidence for an increase](https://voxeu.org/article/effect-age-willingness-take-risks) (at least during working life) and others finding [little increase](https://onlinelibrary.wiley.com/doi/abs/10.1016/j.rfe.2003.09.010). It is plausible, though, that investment advisors have insight that is hard to extract from statistical patterns but easy to perceive in interactions between live human beings.  (New research suggests that any increases in risk aversion among older people reflect [cognitive decline](https://www.nature.com/articles/ncomms13822) associated with reduced ability to process information).
#
# For technical reasons, it is somewhat difficult to incorporate values of risk aversion that vary directly with age.  But your willingness to invest in risky assets depends on both your degree of aversion to risk and your perception of the size of the risk.  So a backdoor way to examine the consequences of risng risk aversion with age is to assume that the perceived riskiness of stock investments goes up with age.  
#
# That is what is done in the last figure below.  
#
# ![risky_age](figures/figure_risky_age/RShare_Means_100_age.png)

# %% [markdown]
# ### Other Experiments
#
# The `PortfolioConsumerType` tool makes it easy to explore other alternatives.  For example, after the CGM paper was published, better estimates [became available](https://doi.org/10.1016/j.jmoneco.2010.04.003) about the degree and types of income uncertainty that consumers face at different ages.  The most important finding was that the degree of uncertainty in earnings is quite large for people in their 20s but falls sharply then flattens out at older ages.  
#
# It seems plausible that this greater uncertainty in labor earnings could account for the fact that in empirical data young people have a low share of their portfolio invested in risky assets; economic theory says that an increase in labor income risk should [reduce your willingness to expose yourself to financial risk](https://www.jstor.org/stable/2951719).
#
# But the figure below shows that even when we update the model to incorporate the improved estimates of labor income uncertainty, the model still says that young people should have 100 percent of their savings in the risky asset.
#
# ![Parameters_1940s_shocks](figures/Figure_Parameters_1940s_shocks/RShare_Means.png)
#
# Many other experiments are possible in the framework, but the conclusion is always the same: Even if people expect that stock returns in the future will be substantially lower than they have been in the past, for most people most of the time, the return on stock market investments more than compensates for any reasonable degree of risk aversion.

# %% [markdown]
# #### What Might Still Be Missing
#
# Some experiments are NOT possible in the framework.  Perhaps the most important is that our tool provides no way to take into account the risks entailed in homeownership.  Houses, like stocks, are assets whose price can go up or down.  Since housing wealth constitutes the majority of the wealth of most consumers, the model's failure to take into account the effects that homeownership would have on the optimal choice of risky investment in other (non-housing) forms is a serious enough failing to call into question the soundness of its conclusions.  
#
# The Think Forward Institute grant that funded this work has a second component:  The addition of a realistic treatment of the effects of home ownership on the optimal share of financial investment in risky assets.  This is a question that is at the frontier of what is possible using the kinds of tools we are developing.  We are interested to see whether a proper treatment of homeownership will be enough to temper the recommendations of the model to invest heavily in other risky assets.  The answer is not clear -- which is why we need a model!

# %% [markdown]
# #### Code
#
# The computer code to reproduce all of the figures in this notebook, and a great many others, can be executed by installing the HARK toolkit and cloning the REMARK repo.  The small unix program `do_all_code.sh` at the root level of the [REMARKs/CGMPortfolio]() directory produces everything.
#
# The full replication of the CGM paper is referenced in a link below.

# %% [markdown]
# #### References
#
# Cocco, J. F., Gomes, F. J., & Maenhout, P. J. (2005). Consumption and portfolio choice over the life cycle. The Review of Financial Studies, 18(2), 491-533.  [doi.org/10.1093/rfs/hhi017](https://doi.org/10.1093/rfs/hhi017)
#
# Velásquez-Giraldo, Mateo and Matthew Zahn.  Replication of Cocco, Gomes, and Maenhout (2005).  [REMARK](https://github.com/econ-ark/REMARK/blob/master/REMARKs/CGMPortfolio/Code/Python/CGMPortfolio.ipynb)
