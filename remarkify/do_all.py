# To generate all of the figures in the paper, execute this script using the command-line command
# ipython do_all.py

import subprocess
import glob

def run_remark():
    """ check for python files in Code/Python directory
        this will print out an error if no .py available
        Jupytext should be used before running this file.
    """
    filename = glob.glob("Code/Python/*.py")
    notebookname = glob.glob("Code/Python/*.ipynb")
    if len(filename) == 0 and len(notebookname) == 0:
        raise ValueError('No python or jupyter notebook found in the Code/Python directory')
    elif len(filename) == 0:
        print(f"Using nbconvert to convert {notebookname[0].split('/')[-1]} to a python file")
        try:
            import nbconvert
        except:
            raise ImportError('nbconvert is needed to convert the notebook into a python file')
        subprocess.run([f"jupyter-nbconvert --to python {notebookname[0]}"], shell=True)
        filename = notebookname[0][:-5] + 'py'
    else:
        filename = filename[0]
    directory = '/'.join(filename.split('/')[:-1])
    python_file = filename.split('/')[-1]
    subprocess.run([f"cd {directory}; ipython {python_file}"], shell=True)

run_remark()
