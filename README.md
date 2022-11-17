# This repository contains the code used in the paper "Positively Weighted Kernel Quadrature via Subsampling" (NeurIPS2022, https://arxiv.org/abs/2107.09597v3)

## Python Files
The python files are written for Pyhon3 and require Gurobi Optimizer (https://www.gurobi.com/). They are divided into three categroires:

- `emp_nys.py`, `emp_mer.py`, `grlp.py` are libraries.
  - `emp_nys.py` provides our main contribution: recombination algorithm with a kernel given by Nyström approximation. You can change the kernel `emp_nys.k` to use it for your own problem. Further tutorials will be added in due course.
  - `emp_mer.py` is using the trigonometric functions, but you can pass it the known eigenvalues/functions of your own problem.
  - `grlp.py` provides LP and CQP functions suitable for using the Gurobi optimizer in our setting.
- Files starting from `experiment` directly run the experiments done in the paper. 
  - `experiment_sobolev.py` and `experiment_data.py` respectively give experiments done in Section 3.1 and 3.2.
  - `experiment_rchq_sobolev.py` and `experimet_rchq_data.py` respectively give experiments done in Section E.1 and E.2.
- The other python files are providing the details of experiments and called in files starting from `experiment`.

## Results Folders
The folders `/results` and `/results_rchq` respectively contain files used in Section 3 and E.
In both folders, you will find there are two different types of filenames:

- `d*s*t50.{pdf,txt}` provide results on periodic Sobolev spaces. The filenames show the dimension, smoothness, and the number of trials for each method in this order. The last value is 50 for all the experiments done in the paper, but you can edit `experiment_*.py` to do faster experiments with smaller `t` values.
Text files provide the average squared worst-case error, its sample standard deviation, and average runtime for each method.
Note that due to the log plot in y-axis, when making the corresponding graphs, we are taking the average and standard deviation of their logarithms, so the values in the text files are not directly used in plotting the results.

- `{3Dnet,PPlant}_Gaussian_t50.{pdf,txt}` provide results for the corresponding ML datasets. In our code we also have the rational quadratic function `RatQuad` as another option than `Gaussian`, but this case is not included in the paper. The description of text files are the same as that of periodic Sobolev spaces.

## ML Datasets: download needed!
To run the experiments on measure reduction in ML datasets, you need to download the datasets from UCI Machine Learning Repository and add them in `/data` folder with the following filenames.
- `3D_spatial_network.txt` from https://archive.ics.uci.edu/ml/machine-learning-databases/00246/.
- `Combined Cycle Power Plant Data Set.txt` from https://archive.ics.uci.edu/ml/machine-learning-databases/00294/.
  - Extract `CCPP.zip` and find `Folds5x2_pp.xlsx`. Then make a named text file using the numbers from `A2` to `E9569` in its sheet 5 using comma separation in each line.
