# SMILES Transformer

## Setup
* Create an anaconda environment: `conda env create -f environment.yml`
* Activate the environment: `conda activate smiles_transformer`

## Train
* Run `bash run.sh` or `python -m smi_transformer.train -e 1 -v data/vocab.pkl -d data/250k_rndm_zinc_drugs_clean_3.csv -o output -n custom -b 1024 -w 8`
* Available arguments: `python -m smi_transformer.train -h`
