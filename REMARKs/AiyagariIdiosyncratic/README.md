1. Start with a Ubuntu Virtual Machine that has Anaconda 3 installed on it. Launch the virtual machine. In a unix shell, execute the commands:

     *     pip install econ-ark
   
     *     git clone https://github.com/zhang13atJHUecon/REMARK.git
   
     *     cd REMARK/REMARKs/AiyagariIdiosyncratic
     
     *     sudo ./doEverything.sh
    
(It takes around 2 minutes to run the 'doEverything.sh')

2. The following graph contains main files and shows how this folder is structured.

   ```mermaid
   graph LR;
      Parent-->Aiyagari1994QJE.ipynb
      Parent-->Aiyagari.yaml
      Parent-->Code.ipynb
      Parent-->Tex
      Parent-->do_all.py
      Parent-->README.md
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
* "doEverything.sh" is a bash shell script which calls "do_all.py" to run. In other words, it runs everything in Linux System. 
   
   
   
   