Hands-On Heterogeneous Agent Macroeconomics Using the [Econ-ARK/HARK](http://econ-ark.org) Toolkit

[Christopher D. Carroll](http://www.econ2.jhu.edu/people/ccarroll)

Syllabus for a short course on Heterogeneous Agent Macroeconomics

Budapest School of Central Bank Studies

July, 2019

Because Representative Agent (‘RA’) models were not useful for understanding much of what happened in the Great Recession, policymakers including Larry [Summers](#XsummersWolf2) ([2011](#XsummersWolf2)), Fed Chair Janet [Yellen](#XyellenHetero) ([2016](#XyellenHetero)), former IMF Chief Economist Olivier [Blanchard](#XblanchardDSGE) ([2016](#XblanchardDSGE)), ECB Governing Board Member Benoit [Coeure](#XcoeureHetero) ([2013](#XcoeureHetero)), and Bank of England Chief Economist Andy [Haldane](#XhaldaneDappled) ([2016](#XhaldaneDappled)) have suggested that incorporation of heterogeneity (for example, across borrowers and lenders) must be an essential part of the agenda in developing new and better models. In confirmation of that intuition, a number of recent papers, most notably [Kaplan, Moll, and Violante](#XkmvHANK) ([2018](#XkmvHANK)) and [Krueger, Mitman, and Perri](#XkmpHandbook) ([2016](#XkmpHandbook)), have developed models that include a realistic description of microeconomic heterogeneity, and have shown that such models can generate more sensible macroeconomic implications than RA models for important questions like the operation of fiscal and monetary policy.

This course will provide a hands-on introduction to the construction of models with ‘serious’ heterogeneity (that is, heterogeneity that matches the microeconomic facts that theory suggests should matter for macroeconomic outcomes like consumption dynamics); why such heterogeneous agent (‘HA’) models have implications different from those of RA models; and how existing HA models can be adapted to new questions. (‘Hands-On’ means that students with their own laptops will run the and experiment with the code that solves these models in class.)

The course will have two main elements: Lectures explaining the conceptual foundations of the models work; and hands-on demonstrations of live working versions of such models using the open-source [Econ-ARK/HARK](http://econ-ark.org/HARK) toolkit.

Students should bring a laptop on which they have permissions to install and run new software. Prior to class, on that laptop, students should have installed the [anaconda3](https://www.anaconda.com/what-is-anaconda/) stack, which is a distribution of python 3 that includes a robust set of extra tools that are useful for doing computational work. A good guide to installing anaconda is [here](https://github.com/mmcky/nyu-econ-370/blob/master/install-local-guide.pdf).

(In “readings” below, starred readings are strongly suggested)

### 1  Preliminaries

I have hired a team of people from [Alphacruncher](https://alphacruncher.com) to help with some of the technical and communications setup for the course, and a team member will be present at the beginning of the first day of classes, so we will begin the “hands-on” aspect of the coruse from the very beginning.

To minimize problems that can crop up with differences across computing platforms and security, we will be doing most of the work for the class “in the cloud” – using a virtual machine configuration set up by Alphacruncher.

But everything we do should be replicable by students on their own laptops if they install HARK and the free “Anaconda” python stack (set of compuataional tools).

1.  Install [Anaconda](https://docs.anaconda.com/anaconda/install): [https://docs.anaconda.com/anaconda/install](https://docs.anaconda.com/anaconda/install)
2.  Get Git
    -   [Get the command-line tool](https://atlassian.com/git/tutorials/install-git): [https://atlassian.com/git/tutorials/install-git](https://atlassian.com/git/tutorials/install-git)
    -   [Get a GitHub Account](https://github.com/join)
    -   [Download the GitHub Desktop App](https://desktop.github.com)
        -   And connect it to your online GitHub account
3.  [Install HARK](https://github.com/https://github.com/econ-ark/HARK#Installing-Hark): Go to “Quick Start” in the README.md
    -   Follow the instructions for installing HARK for Anaconda
4.  Clone the [DemARK](https://github.com/econ-ark/DemARK) and [REMARK](https://github.com/econ-ark/REMARK) repos
    -   git clone https://github.com/econ-ark/DemARK
    -   git clone https://github.com/econ-ark/REMARK
5.  Using python from the command line:
    -   pip install nose
    -   python -c import HARK ; print(HARK.\_\_file\_\_)
    -   cd \[root directory for HARK\]
    -   python -m nose

### 2  Motivation

Models with serious microfoundations yield fundamentally different conclusions than RA models about core questions in macroeconomics.

1.  How monetary policy works
    -   HA channels account for most of the mechanism of monetary transmission
2.  Whether fiscal policy works
    -   ‘serious’ HA models are consistent with evidence of MPC’s of 0.5
3.  What made the Great Recession Great
    -   RA models: Mostly a supply shock
    -   HA models: Mostly a demand shock

Slides:

-   [Intro to Monetary Policy with Heterogeneity](https://github.com/llorracc/resources/blob/master/Slides/CrawleyMonPolicywithHeterogeniety.pdf), [Crawley](#XCrawleyMonPolicywithHeterogeneity) ([2019](#XCrawleyMonPolicywithHeterogeneity))

Readings:

-   [Ahn et al](http://www.princeton.edu/~moll/WIMM.pdf) ([2017](#XakmwwInequality)), Introduction, Conclusion
    -   Compact and well written discussion of the state and progress of HA macro.
-   [Carroll and Crawley](#XakmwwInequality-Discuss) ([2017](#XakmwwInequality-Discuss)), [Sections 1, 2, and 4](http://econ.jhu.edu/people/ccarroll/discuss/2017-04_NBER_Macro-Annual/akmwwInequality/)
    -   This discussion of that paper puts the relationship of HA to RA models in context.

### 3  Micro Models

#### 3.1  Micro Consumption Theory Refresher

The course will assume that students are familiar with standard quantitative tools for solving RA models, like DYNARE. The bulk of the “hands-on” part of the course will therefore involve learning and using tools for solving micro problems with ‘serious’ microfoundations.

##### 3.1.1  The Infinite Horizon Perfect Foresight Model

Absolute, Return, and Growth Impatience

Notes:

-   [Consumption Under Perfect Foresight and CRRA Utility](http://econ.jhu.edu/people/ccarroll/public/LectureNotes/Consumption/PerfForesightCRRA/)
-   [The Certainty Equivalent Consumption Function](http://econ.jhu.edu/people/ccarroll/public/LectureNotes/Consumption/ConsumptionFunction/)

##### 3.1.2  Consumption With Labor Income Uncertainty

-   Notes:    [A Tractable Model of Buffer Stock Saving](http://econ.jhu.edu/people/ccarroll/public/LectureNotes/Consumption/TractableBufferStock/)
-   Notebook: [Interactive Demo](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks/TractableBufferStockQuickDemo.ipynb)

##### 3.1.3  Rate-Of-Return Uncertainty without Labor Income

Under CRRA utility, without labor income risk:

1.  The consumption function is linear
2.  An increase in risk reduces consumption and the MPC

Notes: [Consumption out of Risky Assets](http://econ.jhu.edu/people/ccarroll/public/LectureNotes/Consumption/CRRA-RateRisk/)

[Consumption With Portfolio Choice](http://econ.jhu.edu/people/ccarroll/public/LectureNotes/AssetPricing/C-With-Optimal-Portfolio/)

Origins: [Merton](#Xmerton:restat) ([1969](#Xmerton:restat)), [Samuelson](#Xsamuelson:portfolio) ([1969](#Xsamuelson:portfolio))

##### 3.1.4  Habits

Notes:

-   [Consumption Models with Habit Formation](http://econ.jhu.edu/people/ccarroll/public/LectureNotes/Consumption/Habits/)

### 4  Computational Tools

#### 4.1  Vision for the Econ-ARK Project

-   [Intro-To-Econ-ARK](https://github.com/econ-ark/PARK/blob/master/Intro-To-Econ-ARK-Overlay.pdf)

### 5  Hands-On Introduction

Here we will explain how to begin using the [Econ-ARK](http://econ-ark.org) toolkit for heterogeneous agent macro modeling. Students will be taught how to use the toolkit to solve increasingly sophisticated models, starting with partial equilibrium perfect foresight models and ending with some exercises using a full general equilibrium micro-macro model with idiosyncratic and aggregate risks.

#### 5.1  A Gentle Introduction

This section builds our first simple models using the toolkit

##### 5.1.1  Perfect Foresight

Notebook: [A Gentle Introduction to HARK - Perfect Foresight](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks/Gentle-Intro-To-HARK-PerfForesightCRRA.ipynb)

##### 5.1.2  Adding ‘Serious’ Income Uncertainty

Notebook: [A Gentle Introduction to Buffer Stock Saving](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks/Gentle-Intro-To-HARK-Buffer-Stock-Model.ipynb)

#### 5.2  Liquidity Constraints, Precautionary Saving, and Impatience

1.  The Growth Impatience Condition
2.  Liquidity Constraints and Precautionary Saving
3.  Impatience and Target Wealth

Notebook: [BufferStockTheory Problems](https://next.datahub.ac/open/13/REMARK/REMARKs/BufferStockTheory/BufferStockTheory-Problems.ipynb)

#### 5.3  ‘Serious’ Wealth Inequality

Notebook: [Micro-and-Macro-Implications-of-Very-Impatient-HHs-Problems](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs-Problems.ipynb) References: [Carroll, Slacalek, Tokuoka, and White](#XcstwMPC) ([2017](#XcstwMPC))

#### 5.4  Matching the Distribution – of the MPC

-   : [Figure 5,](http://www.econ2.jhu.edu/people/ccarroll/papers/cstwMPC/#x1-130075) [Carroll, Slacalek, Tokuoka, and White](#XcstwMPC) ([2017](#XcstwMPC))
-   : [Figure 10b,](https://github.com/llorracc/Figures/blob/master/Crawley-MPC-By-Liquid-Assets.png) [Crawley and Kuchler](#XckConsumption) ([2018](#XckConsumption))

#### 5.5  Hands-On with Real HA Models

For an economy in steady state (that is, with constant factor prices like interest rates and wages), models with ‘serious’ income heterogeneity have been solvable in partial equilibrium since about 1990 ([Zeldes](#XzeldesStochastic) ([1989](#XzeldesStochastic)), [Deaton](#XdeatonLiqConstr) ([1991](#XdeatonLiqConstr))). Calculating an equilibrium distribution of wealth that results from those policy functions and matching it to the total amount of observed wealth (and a corresponding interest rate) was first done by [Hubbard, Skinner, and Zeldes](#Xhsz:importance) ([1994](#Xhsz:importance)) using a supercomputer. [Aiyagari](#Xaiyagari:ge) ([1994](#Xaiyagari:ge)) proposed a radically simple model that did not attempt to match the distributions of wealth and income, but could be solved without a supercomputer.

In a rational expectations steady state, there are no expected changes in interest rates, wages, or the distribution. Aggregate fluctuations make calculation of an RE equilibrium massively more difficult, because:

1.  Meaningful aggregate fluctuations will change the distribution of wealth and income
2.  The amount of aggregate saving depends on how aggregate wealth and income are distributed
3.  The amount of saving determines future factor prices
4.  In principle, RE therefore requires that everyone know the entire distribution of wealth, income, and any other state variables in the population

The problem therefore suffers from a severe case of the “curse of dimensionality.” (That is, it’s really hard!). The first paper to tackle the problem was [Krusell and Smith](#XksHetero) ([1998](#XksHetero)). Work by [Bayer and Luetticke](#XblSolving) ([2018](#XblSolving)) builds on all of the prior work to construct a reasonable HANK model that can be solved in a few minutes on a laptop. The key contribution of [Krusell and Smith](#XksHetero) ([1998](#XksHetero)) was to discover that, in practice, highly accurate predictions of future aggregate states could be made using only the mean of the current aggregate capital stock

Notebook: [KrusellSmith.ipynb](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks/KrusellSmith.ipynb)

#### 5.6  The Micro Steady State and Macro Fluctuations

A problem with solving methods using the original Krusell Smith method is that the computational challenge was so great that only the simplest such models could be solved, and the ability to construct standard tools like impulse response functions to aggregate shocks was very limited.

[Reiter](#XreiterSolving) ([2009](#XreiterSolving)) showed how to solve such problems several orders of magnitude faster; the essence of his idea was to solve the micro problem for the steady-state distribution, and then capture business cycle fluctuations by figuring out how to perturb the decision rules and the distribution appropriately.

Building on his work, the last few years have seen great further strides in speed and power of such tools.

References:

-   [Reiter](#XreiterSolving) ([2009](#XreiterSolving))
-   [Boppart, Krusell, and Mitman](#XbmpMITshocks) ([2018](#XbmpMITshocks))
-   [Ahn, Kaplan, Moll, Winberry, and Wolf](#XakmwwInequality) ([2017](#XakmwwInequality))
-   [Bayer and Luetticke](#XblSolving) ([2018](#XblSolving))

#### 5.7  The Bayer-Luetticke Method

-   Notebooks:
    -   [OneAsset HANK Model](https://next.datahub.ac/open/13/HARK/BayerLuetticke/notebooks/OneAsset-HANK.ipynb)
    -   [TwoAsset HANK Model](https://next.datahub.ac/open/13/HARK/BayerLuetticke/notebooks/TwoAsset.ipynb)
    -   [DCT-Copula-Illustration](https://next.datahub.ac/open/13/HARK/BayerLuetticke/notebooks/DCT-Copula-Illustration.ipynb)

#### 5.8  Other Literature

References:

-   [Monetary Policy Transmission with Many Agents](https://github.com/llorracc/Resources/blob/master/Papers/SSinHANK.pdf), [Crawley and Lee](#XSSinHANK) ([2019](#XSSinHANK))
-   [Macroprudential Policies in a Heterogeneous Agent Model of Housing Default](https://pdfs.semanticscholar.org/8e9d/dfe7c204bbfa8a23f42f4931461fb467fc08.pdf?_ga=2.95712860.1156899890.1563925023-1991616136.1563925023), [Khan](#XkhanMacroPru) ([2019](#XkhanMacroPru))
-   [Redistribution, risk premia, and the macroeconomy](https://github.com/llorracc/resources/blob/master/Slides/klRiskPremia.pdf), [Kekre and Lenel](#XklRiskPremia) ([2019](#XklRiskPremia))
-   [The Missing Intercept: A Sufficient Statistics Approach to General Equilibrium Effects](https://github.com/llorracc/resources/blob/master/Slides/wolfGE-invariance.pdf), [Wolf](#XwolfGEInvariance) ([2019](#XwolfGEInvariance))

### References

   ahn, sehyoun, greg kaplan, benjamin moll, thomas winberry, and christian wolf (2017): “When Inequality Matters for Macro and Macro Matters for Inequality,” NBER Macroeconomics Annual, 32.

   aiyagari, s. rao (1994): “Uninsured Idiosyncratic Risk and Aggregate Saving,” Quarterly Journal of Economics, 109, 659–684.

   bayer, christian, and ralph luetticke (2018): “Solving Heterogeneous Agent Models In Discrete Time With Many Idiosyncratic States By Perturbation Methods,” Centre for Economic Policy Research, Discussion Paper 13071.

   blanchard, olivier (2016): “Do DSGE Models Have a Future?,” Discussion paper, Petersen Institute for International Economics, Available at <https://piie.com/system/files/documents/pb16-11.pdf>.

   boppart, timo, per krusell, and kurt mitman (2018): “Exploiting MIT Shocks in Heterogeneous-Agent Economies: The Impulse Response as a Numerical Derivative,” Journal of Economic Dynamics and Control, 89(C), 68–92.

   carroll, christopher d., and edmund crawley (2017): “Discussion of ‘When Inequality Matters for Macro and Macro Matters for Inequality’,” Discussion paper, NBER.

   carroll, christopher d., jiri slacalek, kiichi tokuoka, and matthew n. white (2017): “The Distribution of Wealth and the Marginal Propensity to Consume,” Quantitative Economics, 8, 977–1020, At [http://econ.jhu.edu/people/ccarroll/papers/cstwMPC](http://econ.jhu.edu/people/ccarroll/papers/cstwMPC).

   coeure, benoit (2013): “The relevance of household-level data for monetary policy and financial stability analysis,” .

   crawley, edmund (2019): “Intro to Monetary Policy with Heterogeneity,” Slides Presented at JHU “Computational Methods in Economics”.

   crawley, edmund, and andreas kuchler (2018): “Consumption Heterogeneity: Micro Drivers and Macro Implications,” working paper 129, Danmarks Nationalbank.

   crawley, edmund, and seungcheol lee (2019): “Monetary Policy Transmission with Many Agents,” Manuscript, Johns Hopkins University.

   deaton, angus s. (1991): “Saving and Liquidity Constraints,” Econometrica, 59, 1221–1248, [http://www.jstor.org/stable/2938366](http://www.jstor.org/stable/2938366).

   haldane, andy (2016): “The Dappled World,” Discussion paper, Bank of England, Available at <http://www.bankofengland.co.uk/publications/Pages/speeches/2016/937.aspx>.

   hubbard, r. glenn, jonathan s. skinner, and stephen p. zeldes (1994): “The Importance of Precautionary Motives for Explaining Individual and Aggregate Saving,” in The Carnegie-Rochester Conference Series on Public Policy, ed. by Allan H. Meltzer, and Charles I. Plosser, vol. 40, pp. 59–126.

   kaplan, greg, benjamin moll, and giovanni l. violante (2018): “Monetary Policy According to HANK,” American Economic Review, 108(3), 697–743.

   kekre, rohan, and moritz lenel (2019): “Redistribution, risk premia, and the macroeconomy,” Slides Presented at NBER ‘Micro to Macro’ Working Group.

   khan, shujaat (2019): “Macroprudential Policies in a Heterogeneous Agent Model of Housing Default,” Department of Economics, Johns Hopkins University.

   krueger, dirk, kurt mitman, and fabrizio perri (2016): “Macroeconomics and Household Heterogeneity,” Handbook of Macroeconomics, 2, 843–921.

   krusell, per, and anthony a. smith (1998): “Income and Wealth Heterogeneity in the Macroeconomy,” Journal of Political Economy, 106(5), 867–896.

   merton, robert c. (1969): “Lifetime Portfolio Selection under Uncertainty: The Continuous Time Case,” Review of Economics and Statistics, 51, 247–257.

   reiter, michael (2009): “Solving heterogeneous-agent models by projection and perturbation,” Journal of Economic Dynamics and Control, 33(3), 649–665.

   samuelson, paul a. (1969): “Lifetime Portfolio Selection by Dynamic Stochastic Programming,” Review of Economics and Statistics, 51, 239–46.

   summers, lawrence h. (2011): “Larry Summers and Martin Wolf on New Economic Thinking,” Financial Times interview, [http://larrysummers.com/commentary/speeches/brenton-woods-speech/](http://larrysummers.com/commentary/speeches/brenton-woods-speech/).

   wolf, christian (2019): “The Missing Intercept: A Sufficient Statistics Approach to General Equilibrium Effects,” Slides Presented at NBER ‘Micro to Macro’ Working Group.

   yellen, janet (2016): “Macroeconomic Research After the Crisis,” Available at <https://www.federalreserve.gov/newsevents/speech/yellen20161014a.htm>.

   zeldes, stephen p. (1989): “Optimal Consumption with Stochastic Income: Deviations from Certainty Equivalence,” Quarterly Journal of Economics, 104(2), 275–298.
