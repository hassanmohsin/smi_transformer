#!/bin/bash
# train
python -m transformer.train --params params.json --batch_size 1024 --n_worker 56
