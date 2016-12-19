2D/3D SC Decoding

model_to_metric was getting pretty big, and starting to contain some things 
which have nothing to do with the metric.

Main Reposotory:
git clone git@gitlab.com:bcriger/sc_decoding.git

Pre-requistes:

Sparse Pauli : 
git clone https://github.com/bcriger/sparse_pauli.git

Circuit Metric:
git clone git@gitlab.com:bcriger/circuit_metric.git

QUAEC:
git clone https://github.com/cgranade/python-quaec.git

To setup these packages, run the following:
Python 2: python2 setup.py install --user
Python 2: python3 setup.py install --user

if some packages are reported missing, you can install them by pip by:
Python 2: pip2 install networkx --user
Python 3: pip3 install networkx --user

Usage:
Simple test scripts are available as decoding_2d_test.py and run_script_par.py. These can
be executed by:

python decoding_2d_test.py
python run_script_par.py
