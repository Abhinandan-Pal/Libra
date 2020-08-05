# Libra

<p align="center">
  <img src ="https://raw.githubusercontent.com/caterinaurban/Libra/master/icon.png" width="25%"/>
</p>

Nowadays, machine-learned software plays an increasingly important role 
in critical decision-making in our social, economic, and civic lives.

Libra is a static analyzer for certifying **fairness** of *feed-forward neural networks* 
used for classification of tabular data. Specifically, 
given a choice (e.g., driven by a causal model) of input features 
that are considered (directly or indirectly) sensitive to bias,
a neural network is fair if the classification 
is not affected by different values of the chosen features. 

When certification succeeds, Libra provides definite guarantees, 
otherwise, it describes and quantifies the biased behavior.

Libra was developed to implement and test the analysis method described in:

	C. Urban, M. Christakis, V. Wüstholz, F. Zhang - Perfectly Parallel Fairness Certification of Neural Networks
	Contidionally accepted to appear in Proceedings of the ACM on Programming Languages (OOPSLA), 2020.

## Getting Started 

### Prerequisites

* Install **Git**

* Install [**APRON**](https://github.com/antoinemine/apron)

    * Install [**GMP**](https://gmplib.org/) and [**MPFR**](https://www.mpfr.org/)
    
    | Linux                              | Mac OS X                                   |
    |------------------------------------| ------------------------------------------ |
    | `sudo apt-get install libgmp-dev`  | `brew install gmp`                         |
    |                                    | `ln -s /usr/local/Cellar/gmp/ /usr/local/` |
    |                                    |                                            |
    | `sudo apt-get install libmpfr-dev` | `brew install mpfr`                        |
    |                                    | `ln -s /usr/local/Cellar/mpfr /usr/local/` |

    * Install **APRON**
    
    | Linux or Mac OS X                                    |
    | ---------------------------------------------------- |
    | `git clone https://github.com/antoinemine/apron.git` |
    | `cd apron`                                           |
    | `./configure -no-cxx -no-java -no-ocaml -no-ppl`     |
    | `make`                                               |
    | `sudo make install`                                  |


* Install [**Python 3.7**](http://www.python.org/)

* Install ``virtualenv``:

    | Linux or Mac OS X                     |
    | ------------------------------------- |
    | `python3.7 -m pip install virtualenv` |

### Installation

* Create a virtual Python environment:

    | Linux or Mac OS X                     |
    | ------------------------------------- |
    | `virtualenv --python=python3.7 <env>` |
    
* Install Libra in the virtual environment:

    * Installation from local file system folder 
    (e.g., obtained with `git clone https://github.com/caterinaurban/Libra.git`): 
      
      | Linux or Mac OS X                                  |
      | -------------------------------------------------- |
      | `./<env>/bin/pip install <path to Libra's folder>` |
      
    or, alternatively:

    * Installation from GitHub:

      | Linux or Mac OS X                                                        |
      | ------------------------------------------------------------------------ |
      | `./<env>/bin/pip install git+https://github.com/caterinaurban/Libra.git` |

      A specific commit hash can be optionally specified by appending `@<hash>` to the command.

### Command Line Usage

Libra expects as input a *ReLU-based feed-forward neural network* in Python program format.
This can be obtained from a Keras model using the script `keras2python.py` (within Libra's `src/libra/` folder) as follows:

  | Linux or Mac OS X                      |
  | -------------------------------------- |
  | `python3.7 keras2python.py <model>.h5` |
   
The script will produce the corresponding `<model>.py` file. 
In the file, the inputs are named `x00`, `x01`, `x02`, etc. 

A *specification* of the input features is also necessary for the analysis.
This has the following format, 
depending on whether the chosen sensitive feature for the analysis 
is categorical or continuous:

  | Categorical | Continuous |
  | ----------- | ---------- |
  | `number of inputs representing the sensitive feature` | `1` |
  | `list of the inputs, one per line` | `value at which to split the range of the sensitive feature` |

The rest of the file should specify the other (non-sensitive) categorical features. 
The (non-sensitive) features left unspecified are assumed to be continuous. 

For instance, these are two examples of valid specification files:

  | Categorical | Continuous |
  | ----------- | ---------- |
  | 2           | 1          |
  | x03         | x00        |
  | x04         | 0.5        |
  | 2           | 2          |
  | x00         | x01        |
  | x01         | x02        |

In the case on the left there is one unspecified non-sensitive continuous feature (`x02`). 

To analyze a specific neural network  run:

   | Linux or Mac OS X                             |
   | --------------------------------------------- |
   | `./<env>/bin/libra <specification> <neural-network>.py [OPTIONS]` | 

The following command line options are recognized:

    --domain [ABSTRACT DOMAIN]
    
        Sets the abstract domain to be used for the forward pre-analysis.
        Possible options for [ABSTRACT DOMAIN] are:
        * boxes (interval abstract domain)
        * symbolic (combination of interval abstract domain with symbolic constant propagation [Li et al. 2019])
        * deeppoly (deeppoly abstract domain [Singh et al. 2019]]) 
        Default: symbolic

    --lower [LOWER BOUND]
    
        Sets the lower bound for the forward pre-analysis.
        Default: 0.25
        
    --upper [UPPER BOUND]
    
        Sets the upper bound for the forward pre-analysis.
        Default: 2
    
    --cpu [CPUs]
    
        Sets the number of CPUs to be used for the analysis.
        Default: the value returned by cpu_count() 

During the analysis, Libra prints on standard output 
which regions of the input space are certified to be fair,
which regions are found to be biased, 
and which regions are instead excluded from the analysis due to budget constraints.

The analysis of the running example from the paper can be run as follows (from within Libra's `src/libra/` folder):

     <path to env>/bin/libra tests/toy.txt tests/toy.py --domain boxes --lower 0.25 --upper 2

Another small example can be run as follows (again from within Libra's `src/libra/` folder):

     <path to env>/bin/libra tests/example.txt tests/example.py --domain boxes --lower 0.015625 --upper 4

The `tests/example.py` file represents a small neural network with three inputs representing two input features 
(one, represented by `x`, is continuous and one, represented by `y0` and `y1`, is categorical). 
The specification `tests/example.txt` tells the analysis to consider the categorical feature sensitive to bias.
In this case the analysis should be able to certify 23.4375% of the input space, 
find bias in 71.875% of the input space, and leave 4.6875% of the input space unanalyzed.
Changing the domain to `symbolic` or `deeppoly` should analyze the entire input space finding bias in 73.44797685362308% of it.
The input regions in which bias is found are reported on standard output. 

## Step-by-Step Experiment Reproducibility

### RQ1: Detecting Seeded Bias

The results of the experimental evaluation performed to answer RQ1 are shown in Tables 7-9 
and summarized in Table 1. To reproduce them one can use the script `german.sh` 
within Libra's `src/libra/` folder. This expects the full path to Libra's executable as input:

    ./german.sh <path to env>/bin/libra

The script will generate... 

TODO

## Authors

* **Caterina Urban**, INRIA & École Normale Supérieure, Paris, France
