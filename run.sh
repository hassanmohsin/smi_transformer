# train
python -m smi_transformer.train -e 1 -v data/vocab.pkl -d data/250k_rndm_zinc_drugs_clean_3.csv -o output -n custom -b 1024 -w 8

