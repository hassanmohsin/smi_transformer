#!/bin/bash
# train
python -m transformer.train --params exps/params_1.json --epochs 50 --batch_size 1024 --n_worker 56
