To run all of the replication code you need to open the file:

    StickyE_MAIN.py

...and change the following code flags to "True" -- however see the Important Note section below before executing: 

    # Choose which models to do work for
    do_SOE  = True
    do_DSGE = True
    do_RA   = True
    
    # Choose what kind of work to do for each model
    run_models = True       # Whether to solve models and generate new simulated data
    calc_micro_stats = True # Whether to calculate microeconomic statistics (only matters when run_models is True)
    make_tables = True      # Whether to make LaTeX tables in the /Tables folder
    make_emp_table = False   # Whether to run regressions for the U.S. empirical table (automatically done in Stata)
    make_histogram = False   # Whether to construct the histogram of "habit" parameter estimates (automatically done in Stata)
    use_stata = False        # Whether to use Stata to run the simulated time series regressions
    save_data = True        # Whether to save data for use in Stata (as a tab-delimited text file)
    run_ucost_vs_pi = False  # Whether to run an exercise that finds the cost of stickiness as it varies with update probability
    run_value_vs_aggvar = False # Whether to run an exercise to find value at birth vs variance of aggregate permanent shocks


Important Note
--------------

There are three main types of model represented by these three true-false flags: 

    do_RA   :  Representative Agent
    do_SOE  :  Small Open Economy
    do_DSGE :  Dynamic Stochastic General Equilibrium

These models can be run independently from one another, and each will take vary different amounts of time and RAM to run: 

    do_RA   :  low resource model:          it runs in approximately 5-10 seconds on an 8GB machine
    do_SOE  :  medium/high resource model:  it runs in approximately 5-15 minutes on a 128GB machine; minimum recommended RAM is 64GB.
    do_DSGE :  high resource model:         it runs in approximately 12-24 hours on a 128GB machine; minimum recommended RAM is 64GB.


Since these models can be run independently of one another, you can set only "do_RA=True" for example, to see a very quicky replication.


Running the Replication
-----------------------

Once you have set the desired execution flags to "True" you can run the code using the following command, and see results in "Results:"

    python StickyE_MAIN.py

