The files in this directory are used to construct the index at [econ-ark.org/materials](http://econ-ark.org/materals)

The mandatory content of this directory consists of:

Two unix shell scripts:
	`do_all_code.sh`: produces all of the computational results of the REMARK
	`doEverything.sh`: unix script that
		1. Executes do_all_code.sh
		2. Does anything else required to assemble the contents of Paper-REMARKS
		   * Like, compiles the paper with `pdflatex` 

and, if the 'handle' of the REMARK (that is, the name of the parent directory of this README file) is [name]

* [name]-One-Sentence-Summary.md: will appear as a brief description in the index. Keep it short!
* [name]-Original-Paper-Title.md: self explanatory; but keep even a long title on one line
* [name]-Original-Paper.bib: BibTeX bibliographical entry for the paper 
* [name]-Original-Paper.url: The DOI or other preferred web location of the paper
* [name].ipynb: Jupyter notebook presenting an overview of the paper 
* [name]-Paper-Original: A 'link' (submodule) to the original paper's GitHub presence (if any)

Optional material (if it exists, it should follow the structure below):

* [name]-Paper-REMARKs: Directory containing any further expositional material.
  * Files:
	* [name]-REMARKs.tex: LaTeX document containing any extra remarks
	* [name]-Extras.ipynb: Jupyter notebook containing anything not in the main notebook
  * Subdirectories:
    * Figures
    * Code/Python
    * Code/[other language]
    * Tables
    * Equations
	* Appendices
	* Data

