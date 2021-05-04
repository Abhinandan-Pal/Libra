# Tool (tool-paper #16)

Nowadays, machine-learned software plays an increasingly important role 
in critical decision-making in our social, economic, and civic lives.

Tool is a static analyzer for certifying **fairness** of *feed-forward neural networks* 
used for classification of tabular data. Specifically, 
given a choice (e.g., driven by a causal model) of input features 
that are considered (directly or indirectly) sensitive to bias,
a neural network is fair if the classification 
is not affected by different values of the chosen features. 

When certification succeeds, Tool provides definite guarantees, 
otherwise, it describes and quantifies the biased behavior.

A preliminary version of Tool was developed to implement and test the analysis method described in:

	C. Urban, M. Christakis, V. Wüstholz, F. Zhang - Perfectly Parallel Fairness Certification of Neural Networks
	In Proceedings of the ACM on Programming Languages (OOPSLA), 2020.

Tool now additionally includes new abstract domains including a generic reduced product domain construction, 
a new auto-tuning mechanism for finding the optimal configuration for Tool’s forward pre-analysis, and
a tasks scheduling optimization to leverage all the available CPUs for Tool’s backward analysis. 
These new features are described in:
	
	Anonynous - Reduced Products of Abstract Domains for Fairness Certification of Neural Networks
	Submitted to SAS 2021, tool-paper #16.

## Getting Started 

### Docker Container

The provided Docker container is a virtual image ready to run Tool and replicate the experiments presented in the SAS submission. We suggest reviewers to run the container using `bash` as follows

    > docker exec -it tool16 bash

Otherwise, without bash the arrow navigators are not available.
Additionally, we already installed `nano` and `vim` inside the container to inspect and modify files.

The container is organized as follow:
```
/home                               # tool repository
├── src                         
│   └── tool
│       ├── abstract_domains        # code
│       ├── core                    # code
│       ├── engine                  # code
│       ├── frontend                # code
│       ├── semantics               # code 
│       ├── tests
│       │   ├── census
│       │   │   ├── logs1           # logs for experiment A
│       │   │   │   └── fetch.py    # log fetch script
│       │   │   ├── logs2           # logs for experiment B
│       │   │   │   └── fetch.py    # log fetch script
│       │   │   ├── logs3           # logs for experiment C
│       │   │   │   └── fetch.py    # log fetch script
│       │   │   ├── 10.py           # network with 10 ReLU
│       │   │   ├── 12.py           # network with 12 ReLU
│       │   │   ├── 20.py           # network with 20 ReLU
│       │   │   ├── 40.py           # network with 40 ReLU
│       │   │   ├── 45.py           # network with 45 ReLU
│       │   │   ├── census.txt      # experiments specification file
│       │   │   ├── census.csv      # original dataset
│       │   │   └── ...
│       │   ├── example.py          # simple example network
│       │   ├── example.txt     
│       │   ├── toy.py              # another simple example network
│       │   ├── toy.txt
│       │   └── ...
│       ├── configurations.sh       # experiment B
│       ├── cpus.sh                 # experiment C
│       ├── keras2python.py         # keras to python conversion script
│       ├── models.sh               # experiment A
│       └── ...
├── README.md                       # this file
└── ...
```

### Command Line Usage

Tool expects as input a *ReLU-based feed-forward neural network* in Python program format.
This can be obtained from a Keras model using the script `keras2python.py` (within Tool's `src/tool/` folder) as follows:

    > python3.7 keras2python.py <model>.h5
   
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

    > tool <specification> <neural-network>.py [OPTIONS]

The following command line options are recognized:

    --domain [ABSTRACT DOMAIN]
    
        Sets the abstract domain to be used for the forward pre-analysis.
        Possible options for [ABSTRACT DOMAIN] are:
        * boxes (interval abstract domain)
        * symbolic (combination of interval abstract domain with symbolic constant propagation 
        [Li et al. - Analyzing Deep Neural Networks with Symbolic Propagation: Towards Higher Precision and Faster Verification (SAS 2019)])
        * deeppoly (deeppoly abstract domain 
        [Singh et al. - An Abstract Domain for Certifying Neural Networks (POPL 2019)]) 
        * neurify (neurify symbolic relaxation 
        [Wang et al. - Efficient Formal Safety Analysis of Neural Networks (NeurIPS 2018)]) 
        * boxes_deeppoly (product of boxes and deeppoly)
        * boxes_neurify (product of boxes and neurify)
        * deeppoly_symbolic (product of deeppoly and symbolic)
        * neurify_symbolic (product of neurify and symbolic)
        * deeppoly_neurify (product of deeppoly and neurify)
        * boxes_deeppoly_neurify (product of boxes, deeppoly, and neurify)
        * deeppoly_neurify_symbolic (product of deeppoly, neurify, and symbolic)
        Default: symbolic

    --lower [LOWER BOUND]
    
        Sets the lower bound for the forward pre-analysis.
        Default: 0.25
    
    --min_lower [LOWER BOUND]
    
        Sets the minimum lower bound for the (auto-tuning of the) forward pre-analysis.
        Default: the value of the lower bound
        
    --upper [UPPER BOUND]
    
        Sets the upper bound for the forward pre-analysis.
        Default: 2
        
    --max_upper [UPPER BOUND]
    
        Sets the maximum upper bound for the (auto-tuning of the) forward pre-analysis.
        Default: the value of the upper bound
    
    --cpu [CPUs]
    
        Sets the number of CPUs to be used for the analysis.
        Default: the value returned by cpu_count() 

During the analysis, Tool prints on standard output 
which regions of the input space are certified to be fair,
which regions are found to be biased, 
and which regions are instead excluded from the analysis due to budget constraints.

The analysis of the running example from the OOPSLA paper can be run as follows (from within Tool's `src/tool/` folder):

     > tool tests/toy.txt tests/toy.py --domain boxes --lower 0.25 --upper 2

Another small example can be run as follows (again from within Tool's `src/tool/` folder):

     > tool tests/example.txt tests/example.py --domain boxes --lower 0.015625 --upper 4

The `tests/example.py` file represents a small neural network with three inputs representing two input features 
(one, represented by `x`, is continuous and one, represented by `y0` and `y1`, is categorical). 
The specification `tests/example.txt` tells the analysis to consider the categorical feature sensitive to bias.
In this case the analysis should be able to certify 23.4375% of the input space, 
find bias in 71.875% of the input space, and leave 4.6875% of the input space unanalyzed.
Changing the domain to any of the other options should analyze the entire input space finding bias in 73.44797685362308% of it.
The input regions in which bias is found are reported on standard output. 

## Step-by-Step Experiment Reproducibility

The experimental evaluation in the SAS submission was conducted on a machine 
with two 16-core Intel ® Xeon ® 5218 CPU @ 2.4GHz, 192GB of RAM, and running CentOS 7.7. with linux kernel 3.10.0.

### Experiment 1: Effect of Neural Network Structure on Precision and Scalability

To reproduce the results shown in Table 1 one can use the script `models.sh` 
within Tool's `src/tool/` folder.

    > ./models.sh

The script will generate the corresponding log files in Tool's `src/tool/tests/census/logs1`.
These can be manually inspected or a table summary of them can be generated 
using the script `fetch.py` in Tool's `src/tool/tests/census/logs1` folder.

> Please take note of the expected execution times before launching the script. 
On a less powerful machine than that used for our evaluation 
it might be preferable to comment out the most time consuming lines 
from the script before launching it.

### Experiment 2: Scalability-vs-Precision Tradeoff

To reproduce the results shown in Table 2 one can use the script `configurations.sh` 
within Tool's `src/tool/` folder.

    > ./configurations.sh

The script will generate the corresponding log files in Tool's `src/tool/tests/census/logs2`.
These can be manually inspected or a table summary of them can be generated 
using the script `fetch.py` in Tool's `src/tool/tests/census/logs2` folder.

> Please take note of the expected execution times before launching the script. 
On a less powerful machine than that used for our evaluation 
it might be preferable to comment out the most time consuming lines 
from the script before launching it.

### Experiment 3: Leveraging Multiple CPUs

To reproduce the results shown in Table 3 one can use the script `cpus.sh`
within Tool's `src/tool/` folder.

    > ./cpus.sh

The script will generate the corresponding log files in Tool's `src/tool/tests/census/logs3`.
These can be manually inspected or a table summary of them can be generated 
using the script `fetch.py` in Tool's `src/tool/tests/census/logs3` folder.

> Please take note of the expected execution times before launching the script. 
On a less powerful machine than that used for our evaluation 
it might be preferable to comment out the most time consuming lines 
from the script before launching it. Note also that this script requires 64 CPUs, modify the script to fit your available resources before running it.

In the `src/tool/tests/census` folder is also present the original dataset `census.csv` 
as well as the 5 trained neural networks (`10`, `12`, `20`, `40`, `45`).
 
