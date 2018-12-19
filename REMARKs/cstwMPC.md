# cstwMPC is a Reproduction

This is a reproduction of the results in the paper "The Distribution of Wealth and the Marginal Propensity to Consume" 
by Carroll, Slacalek, Tokuoka, and White.

The root directory contains six files:

* `do_min.py` shows the wealth distribution comparison between the models with and without heterogenous impatience. 
* `do_mid.py` estimates the mean and spread of the distribution of the impatience across households and compares the model 
implied wealth distribution with the data.
* `do_lifecycleBetaPoint.py`executes the lifecycle model without heterogenous impatience.
* `do_lifecycleBetaDist.py` executes the lifecycle model with heterogenous impatience
* `do_aggregateBetaPoint.py`executes the model with shocks at the aggregate level and without heterogenous impatience.
* `do_aggregateBetaDist.py` executes the model with shocks at the aggregate level and with heterogenous impatience.
