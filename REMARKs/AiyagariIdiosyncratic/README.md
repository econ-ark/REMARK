1. Please open this notebook in *JupyterLab*. To install it with pip, please conduct:

   * pip install jupyterlab

2. This notebook uses *Dolo* and *Dolark* to replicate Aiyagari(1994). To install Dolark, please follow the steps below: 

   * git clone "dolo", "dolang" and "dolark" to the local directory:

     ​    git clone https://github.com/econforge/dolo.git  
     ​    git clone https://github.com/econforge/dolang.git   
     ​    git clone https://github.com/econforge/dolark.git

   *  go to the local directory and do "pip install -e . " for each of the three. (Three times in total) 

     ​    pip install -e .   

   * open "JupyterLab" and import dolo, dolang, and dolark. 

     ​    import dolo 

     ​    import dolang 

     ​    import dolark 

     

3. This notebook also uses *altair* for plotting graphs. To install it, please do:

   * pip install altair

4. This notebook also uses *tabulate* to generate tables of outputs. In order to run these codes, please also install "tabulate":  

   * pip install tabulate

5. The following graph contains main files and shows how this folder is structured.

   ```mermaid
   graph LR;
      Parent-->Aiyagari1994QJE.ipynb
      Parent-->Aiyagari.yaml
      Parent-->Code.ipynb
      Parent-->Tex
      Parent-->do_all.py
      Parent-->README.md
      Parent--> Table_SaingRate.md
      Parent--> doEverything.sh
     
      Tex--> Figures
      Tex--> Tables
      Tex--> Slides
      Tex--> main.tex
      Tex--> Appendix
   
   ```

   Where:
   
   * "Aiyagari1994QJE" is the Jupyter Notebook file, which includes key features of the paper and python codes implementing to replicate the main results of the paper;
   *  "Aiyagari.yaml" is a yaml file where the model is written; 
   * "Code" includes all python codes for replication in the format of ipynb; 
   * "Tex" is a folder where the .tex file is located. It includes main content, figures, tables, slides, and an appendix as a subfile. 
* "do_all.py"  is the file which contains all codes for solving the model, generating tables and figures, saving the table in  both markdown and LaTeX languages, and re-compiling the LaTeX file each time you run the code.
   * "Table_SavingRate.md" is a markdown table reporting the aggregate saving rates calibrated by us and by Aiyagari(1994). The table is re-written every time you run the code. 
* "doEverything.sh" is a bash shell script which calls "do_all.py" to run. In other words, it runs everything in Linus System. 
   
   
   
   