I was following the example of the REMARK code structure discussed previously, which has this format:

    REMARK/
        Applications/
            SolvingMicroDSOPs/    # This is just from the example you sent, not what is in the script
                Calibration/
                Code/
                Equations/
                Figures/
                Slides/
                Tables/
                do_all.py


As I will describe below, the code in `remarkify.py` will create that file structure inside the StickyE repo. There are a few things it can't do, which the user will need to manually do, and I describe that below as well. 

The attached script will take a user-defined dictionary that describes where all the relevant files are, and then plucks them all up and places them in the right place in a newly-created directory *within* the repo that is self-contained and in the correct format.  I've filled out that dictionary in this file so that it works with StickyE.

I wrote it and tested it minimally with StickyE and it works -- it puts all the code in the right place.


Two limitations, and one new note: 

1.  One thing the script can't do is go into the code directly and change the pathnames in the parameters code. *so* it prints a message that says, "please go do this." 

Fortunately Matt wrote the code so it is super easy to do this: you just go change these lines in "StickyEparams.py" to have the right pathname:

```
calibration_dir = os.path.join(my_file_path, "../../Calibration/Parameters/") # Relative directory for primitive parameter files
tables_dir = os.path.join(my_file_path, "../../Tables/")           # Relative directory for saving tex tables
results_dir = os.path.join(my_file_path, "./Results/")         # Relative directory for saving output files
figures_dir = os.path.join(my_file_path, "../../Figures/")         # Relative directory for saving figures
empirical_dir = os.path.join(my_file_path, "../Empirical/")     # Relative directory with empirical files
```

...basically you change them to this: 

```
calibration_dir = os.path.join(my_file_path, "../Calibration/") # Relative directory for primitive parameter files
tables_dir = os.path.join(my_file_path, "../Tables/")           # Relative directory for saving tex tables
results_dir = os.path.join(my_file_path, "./Results/")         # Relative directory for saving output files
figures_dir = os.path.join(my_file_path, "../Figures/")         # Relative directory for saving figures
empirical_dir = os.path.join(my_file_path, "../Empirical/")     # Relative directory with empirical files
```

2.  The script also can't change anything inside the "do_all" file -- it will run as-is. In the StickyE case, the "do_all" file is simply a text file which *tells* the user how to execute the full replication. The reason this wasn't turned into a "do_all.py" file is that it would take some non-trivial re-working of that code to make it work that way. Probably a days work to get the code itself, and about 50% testing done. I mentioned in the commit message that this was probably not something we want to do last minute. 
This is not addressed by the "REMARK-ify" script, because that is far out of scope of that script. It simply tells the user they need to make that adjustment. 


3.  New note, nothing big:   I realized that we need to have an "Empirical" directory added to the REMARK structure, as is currently in StickyE


Finally,  if you want to test this code yourself, do the following:

    # Clone the StickyE repo:
    git clone https://github.com/llorracc/cAndCwithStickyE

    # save the "remarkify" code in the top level of that repo
    cp ~/Downloads/remarkify PATH-TO-STICKYE-REPO

    # cd over and run the code:
    cd PATH-TO-STICKYE-REPO
    python remarkify.py

    # confirm that the code is in the correct structure:
    cd PATH-TO-STICKYE-REPO/cAndCwithStickyE-REMARK/    # cd into new "stand alone" code
    ls -l   # just look at structure

    # Then manually adjust the code in StickyEparams.py as described in (1) above
    cd PATH-TO-STICKYE-REPO/cAndCwithStickyE-REMARK/Code/
    emacs StickyEparams.py

    # Finally, set the flags to run the "RA" version of the model in StickyE_MAIN.py;
    # you'll need:
    #     do_SOE  = False
    #     do_DSGE = False
    #     do_RA   = False
    # 
    #     run_models = False       # Whether to solve models and generate new simulated data
    # 

    cd PATH-TO-STICKYE-REPO/cAndCwithStickyE-REMARK/Code/
    emacs StickyE_MAIN.py
    python StickyE_MAIN.py

Finally, there are two general "requirements" and "README.txt" added in this directory that can be used with the 'remarkify.py' code.
