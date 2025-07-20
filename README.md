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

## Projects and Publications

Each subfolder in the `projects/` directory corresponds to a published study or ongoing research effort. Below is an overview linking each project to its associated publication:

| Project Folder            | Title                                                                                              | Citation / DOI |
|---------------------------|----------------------------------------------------------------------------------------------------|----------------|
| `ni(I)-dimer/`            | *Machine Learning-Guided Development of Trialkylphosphine Ni(I) Dimers and Applications in Site-Selective Catalysis* | [J. Am. Chem. Soc. 2023, 145, 28, 15414–15424](https://doi.org/10.1021/jacs.3c03403) |
| `ni(I)-co2/`              | *Discovery of Ni(I) Complexes for CO2 Insertion Enabled by a Machine Learning-Computational-Selection Sequence* |[J. Am. Chem. Soc. 2025, XXX, XXX, XXX-XXX](https://doi.org/10.1021/jacs.5c00441) |
