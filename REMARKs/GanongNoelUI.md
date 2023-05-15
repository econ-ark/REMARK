---
# CFF required fields
cff-version: 1.1.0 # required (don't change)
# message: If you use this software, please cite it as below. # optional
authors: # required
  - family-names: "Ganong"
    given-names: "Peter"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
  - family-names: "Noel"
    given-names: "Pascal"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
title: "Consumer Spending during Unemployment: Positive and Normative Implications" # required
abstract: Analysis of Models for "Consumer Spending During Unemployment- Positive and Normative Implications"

# REMARK required fields
remark-version: "1.0" # required - specify version of REMARK standard used
references: # required for replications; optional for reproductions; BibTex data from original paper
  - type: article
    authors: # required
      -
        family-names: "Ganong"
        given-names: "Peter"
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "Noel"
        given-names: "Pascal"
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    title: "Consumer Spending During Unemployment: Positive and Normative Implications" # required
    doi: "https://doi.org/10.1257/aer.20170537" # optional
    date: 2019-07 # required
    publisher: "The American Economic Review"

# Econ-ARK website fields 
github_repo_url: "https://github.com/econ-ark/GanongNoelUI" # required 
remark-name: "GanongNoelUI" # required
title-original-paper: "Consumer Spending During Unemployment: Positive and Normative Implications" # optional 

tags:  # Use the relavent tags
  - REMARK
  - Replication

---
  
  Analysis of Models for "Consumer Spending During Unemployment: Positive and Normative Implications"

By Peter Ganong and Pascal Noel 

We thank Xian Ng for outstanding research assistance. Please send feedback and questions to ganong@uchicago.edu.

## Directory Structure
 1. `code/` - all estimation and analysis code
 2. `input/` -  model estimation targets from JP Morgan Chase Institute (JPMCI) data
 3. `out/` - code output including figures, logfiles, and raw tables
 4. `Parameters/` - environment parameters for estimation and analysis; stored model estimation results
 5. `tables/` - formatted tables

## Files in the `code/` directory

### Master Scripts: do_min.py, do_mid.py, do_all.py
The master scripts in the code directory re-run the code. To call the script from the command line, go to the `code/` directory and enter 'python <filename>'. Note that this project is built on python 2.7  
	1. `do_min.py`: Solves the consumption and job search models using estimated parameters in `Parameters/model_params_main.json` to replicate the plots in the paper.  
	2. `do_mid.py`: `do_min.py` + computes the standard errors on the parameter estimates, and performs the welfare simulations in the paper.  
	3. `do_all.py`: `do_mid.py` + estimates the model parameters again using model targets from JPMCI data.  
To simply replicate the results, it is sufficient to run one of the scripts above. 


### Setup scripts
 1. `setup_estimation_parameters.py` reads in `Parameters/params_ui_reemp.json` to build the estimation environment 
 2.  `build_JPMC_targets.py` builds the model estimation targets from files in `input/` and writes to `Parameters/JPMC_inputs.py`. It only needs to be run once.

### Model Solving, Estimation, and Simulation scripts
 3. `solve_cons.py` - contains a function that takes environment and preference parameters and computes optimal consumption in each period as a function of cash-on-hand. It solves this problem using backwards induction for a finite horizon.
 4. `job_search.py` extends `solve_cons.py`. Takes environment and preference parameters and computes optimal consumption and job search in each period as a function of cash-on-hand. 
 * Agent chooses search effort with an isoelastic cost and with the gains from search equal to V_emp(a) - V_unemp(a).
 * To accomplish the above, we compute a value function which sums over utility in each period. We do not need the value function to compute optimal consumption, but we do need it to compute optimal job search effort.
 5. `sparsity.py` solves the sparse model from Gabaix (2016) in the UI context.
 6. `estimate_models.py` takes environmental parameters and consumption and job search moments and solves for the preference parameters that generate consumption and job search behavior similar to the moments. Relies on the class in  `job_search.py`. By default, solves for the models in the paper one at a time. Can also be used to solve multiple models at once on a cluster.
 7. `agent_history.py` With a given set of environmental and preference parameters, simulates employment histories, consumption behavior, and job search behavior for N agents.

### Plotting and Replication scripts
 8. `model_plotting.py` contains plotting functions. `make_plots.py` uses the `rpy2` package to create some wrappers for making PDF plots in R.
 9. `model_plots.py` produces plots using the estimated preference parameters contained in `Parameters/model_params_main.json` and `Parameters/model_params_sec.json`.
 10. `model_welfare.py` performs the simulations for lifecycle welfare analysis.
 11. `comp_SEs.py` calculates standard errors for estimated models.
 Note: all output is in the `/out/` directory.

### Other files
 * `est_robust_gamma.py` estimates models with different risk aversion parameters.
 * `estimate_models_helper.py` contains helper functions for estimating many models simultaneously on a cluster.
 * `grid_submit.sh` shell script for submitting jobs to a cluster. Edit as necessary.
 * `make_plots.py` contains aesthetic options for plots.
 * `prelim.py` is a helper script for setting up the model environment.
 * `est_models_in/` contains `initial_conditions_master.json` for estimating the models in the paper. csv files in the directory are examples of different initial conditions for model estimation. Convert to JSON for estimation using function in `estimate_models_helper.py` 
* `est_models_out` logs final estimates as JSON files when estimating preference parameters for multiple models at the same time on a cluste. Convert to csv using function in `estimate_models_helper.py` 

## References

Ganong, P., & Noel, P. (2019). Consumer spending during unemployment: Positive and normative implications. American economic review, 109(7), 2383-2424.
