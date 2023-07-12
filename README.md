# AIxChem

The AIxChem framework provides a collection of unsupervised machine learning tools to be used for exploration of chemical space with minimal data.

## Getting started

In order to use the framework, ensure you have all the required dependencies installed. 
The required environment files can be found in the ``requirements/`` directory. To create and activate an environment follow the instructions provided below:


### Using conda (recommended)

```sh
conda env create -f requirements.yml
conda activate aixchem
```

### Using pip 
Ensure you have Python3.10 installed on your system

```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Projects

Code that was used throughout projects can be found in the ``projects/`` folder:

- ``projects/ni(I)-dimer/``:  
Machine Learning-Guided Development of Trialkylphosphine Ni(I) Dimers and Applications in Site-Selective Catalysis (https://doi.org/10.1021/jacs.3c03403)

