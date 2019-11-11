# Installing Software Required for REMARKs

Some (most?) [REMARKs](https://github.com/econ-ark/REMARK/master/blob/REMARKs) can be explored without installing any software on your computer. Files in a REMARK directory ending in the extension .ipynb are [jupyter notebooks](https://jupyter.org) which can be executed by the free online tool [mybinder](https://mybinder.org). The README file in the REMARK's root directory should have a link which, when clicked, will launch a mybinder session that can execute the notebook.

The free mybinder tool has only limited computational capacity, though, so for projects that have substantial 
computational content you will need to run the code locally.

## To execute locally: You need python3, pip, and econ-ark

Most computers come with a distribution of python3 and the `pip` install tool. [See the README file here](https://github.com/econ-ark/HARK) for installation instructions for the python, pip, and the econ-ark

# Jupyter Installation

To run jupyter notebooks locally on your computer:

1. [Install jupyter](https://jupyter.org/install).
2. Clone the `REMARK` repo to the folder of your choice
2. Change to the directory in REMARKs for the REMARK you want to explore
3. Run `pip install -r binder/requirements.txt` to install dependencies
4. Enable notebook extensions.

   **On Linux/macOS:**

   Run `binder/postBuild` in your terminal (at a shell in the binder directory, `./postBuild`)

   **On Windows:**

   Run `binder/postBuild.bat`

5. Run `jupyter notebook` from the `REMARK` root folder. You will be prompted to open a page in your web browser. From there, you will be able to run the notebooks.
6. Run the notebook by clicking the `▶▶` button or choosing `Kernel → Restart & Run All`

## To collaborate with someone ...

If you intend to interact with others using GitHub and Jupyter notebooks, you should 
install the [jupytext](https://towardsdatascience.com/introducing-jupytext-9234fdff6c57) tool.

To deal with the well-known problem that normal jupyter notebooks do not "play nicely" with github version control, we will require interactions on jupyter notebooks to be conducted after the installation of jupytext.  Specifically, you will need to follow the instructions at the link for installing jupytext on your computer, and then need to configure it to use the "percent" format. Over time, we intend to add the necessary metadata to all our jupyter notebooks to make them automatically invoke jupytext when compiled.

