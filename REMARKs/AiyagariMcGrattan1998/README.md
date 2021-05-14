# A starter repository for creating a REMARK.

A REMARK `(R[eplications/eproductions] and Explorations Made using ARK)` is a self-contained project whose computational results should be reproducible indefinitely into the future across platforms (Windows, Mac, Linux, ...).

The minimal requirements of a REMARK, articulated below, reflect the infrastructure needed to keep track of REMARKs and to guarantee their reproducibility.  (No other elements of the ARK toolkit need be used).

This is a starter repository to start building your own REMARKs.  

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/REMARK-template/master?filepath=resources%2Fcode%2Ftemplate-example-notebook.ipynb) Binder link for the main notebook in this repository.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/REMARK-template/master?urlpath=voila%2Frender%2Fresources%2Fdashboard%2Fdashboard.ipynb) Binder link for a voila dashboard in this repository.

A maximalist example, in the form of a complete paper with all computational results, is [BufferStockTheory](https://github.com/econ-ark/BufferStockTheory), corresponding to the paper [Theoretical Foundations of Buffer Stock Saving](https://econ-ark.github.io/BufferStockTheory). This example is one of many possible ways you might structure a REMARK.

REMARKs are instantiated in the form of a git repository, whose directory structure is exemplified in this starter repo:
```
- reproduce.sh : a unix bash script (required)
- requirements.txt : (required, if code in Python; see explanation below)
- README.md : (required; includes links to execute REMARK materials)
- code/ (required)
  - python/ (if python is used)
	  - [files-needed-for-reproduce-script-to-work].py
	  - [jupyter-notebook-with-same-name-as-repo].ipynb
	  - ...
  - name_of_other_language/ (if other language is used)
	  - [files-needed-for-reproduce-script-to-work]
- [REMARKname].tex (if reproduce.sh has code that compiles such a document)
- [REMARKname].pdf (the result of compiling [REMARKname].tex, if it exists)
- latex/ (required if [REMARKname].tex is present)
  - [latex support files necessary for compiling the paper, like .bib, .sty, etc]
- figures/ (optional)
  - [REMARKname].tex expects figures in this directory
  - ....
- equations/  (optional)
  - [REMARKname].tex can create or uses latex eqns in this directory
- slides (optional)
  - [REMARKname]-Slides.tex
  - [supporting files for slides, if any]
- [REMARKname].cff : mandatory CFF file (https://citation-file-format.github.io/)
- [REMARKname-OriginalPaper].bib (if this is a replication or reproduction)
- dashboard (optional)
  - [REMARKname]-dashboard.ipynb
- .....
```
As you can see the structure of the REMARK is largely optional and can be molded as required.

## `bash` script(s)

The only strict requirements are the `reproduce.sh` bash script and the [versioned requirements.txt](https://www.idkrtm.com/what-is-the-python-requirements-txt/) file (or a [Dockerfile](https://docs.docker.com/engine/reference/builder/) in case of non-Python code; see the documentation for [`nbreproduce`](https://github.com/econ-ark/nbreproduce#readme)).

These bash scripts can contain any number of sequential steps, like running a python script, building the latex document, etc. Look at the bash script is this repository for an example.

Optionally, you might want to have multiple bash scripts.  For example, if you have a latex `[REMARKname].tex` it will probably prove convenient to have a
##### `reproduce_textetc.sh`
script just to compile it.  If you do this, your main `reproduce.sh` script could invoke the `reproduce_textetc.sh` script to accomplish the reproduction of the latex document.

In that case, you might also want to have a standalone
##### `reproduce_results.sh`
to reproduce the computational results without the latex.  Your `reproduce.sh` script could then be very simple: It would just run `reproduce_results.sh` and `reproduce_textetc.sh`

If reproducing of the results of the project takes a long time, you are encouraged to have a
##### `reproduce_min.sh`
script that reproduces some interesting a subset of the results in some shorter amount of time (ideally, 5 min or less -- please include some indication of how long the scripts take).

### Accomplishing Reproduction

After installing the requirements described below, all results of the project should be reproducible by executing the following command from the command line in this directory
```
$ nbreproduce reproduce.sh
```
or similarly for any other available script (e.g., `nbreproduce reproduce_min.sh`)

### How to install nbreproduce?

Detailed documentation about `nbreproduce` is at [https://econ-ark.github.io/nbreproduce/](https://econ-ark.github.io/nbreproduce/).

`nbreproduce` is available through PyPI and depends on [Docker](https://www.docker.com/products/docker-desktop).

If you already have Docker installed locally you can install `nbreproduce` using
```
$ pip install nbreproduce # make sure you have Docker installed on your machine.
```

### What is a versioned requirements.txt file?

A requirements file tells the exact dependencies required to run the code. This currently only supports packages available on PyPI.

For example if you use econ-ark, numpy, pandas, seaborn and voila in your REMARK notebooks and code a requirements file that would work right now would look like:
```
econ-ark
numpy
pandas
seaborn
voila
```
But this file misses out an important detail which is absolutely necessary to make sure your code is reproducible across systems and in the future, python package versions. A versioned requirements file would look like:
```
econ-ark==0.10.7
numpy==1.19.1
seaborn==0.10.1
pandas==1.1.1
voila==0.1.22
```

### "Publishing" Your REMARKs

When you have finished creating a REMARK you can request that it be ["Published"](https://github.com/econ-ARK/REMARK/master/REMARK-Submission/README.md) on the Econ-ARK website.

### How to create mybinder links/buttons?

- Go to https://mybinder.org
- Fill up the URL field with the link to the repository (example: https://github.com/econ-ark/REMARK-template)
- In path to notebook, fill up the path to notebook (in this example: `resources/code/template-example-notebook.ipynb`)
- Click on the launch button to start the build
- Click on the "Copy the text below, then paste into your README to show a binder badge" to get the markdown and rST text required to create the mybinder button.
