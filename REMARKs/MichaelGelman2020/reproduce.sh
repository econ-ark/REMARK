# this is required as this requries the latest HARK master to work
python3 -m pip install -U git+https://github.com/econ-ark/hark
python -m pip install -U matplotlib
pip install numpy
pip install pandas

ipython do_all.py
