#!/bin/bash
# train
python -m transformer.train --params exps/params_6.json --batch_size 1024 --n_worker 56
