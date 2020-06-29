# SMILES Transformer

## Setup

- Create an anaconda environment: `conda env create -f environment.yml`
- Activate the environment: `conda activate smiles_transformer`

## Data processing

### Process the SMILES

- `python -m transformer.build_corpus -i data/chembl_24_chemreps.csv -o data/chembl_24_chemreps_processed.csv -s canonical_smiles`

### Build vocabulary

- `python -m transformer.build_vocab -c data/chembl_24_chemreps_processed.csv -o data/vocab.pkl`

## Train

- Run `bash run.sh` or `python -m transformer.train -e 1 -v data/vocab.pkl -d data/chembl_24_chemreps_processed.csv -o output -n custom -b 1024 -w 8`
- Available arguments: `python -m transformer.train -h`
