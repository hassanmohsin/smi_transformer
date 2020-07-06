#!/bin/bash
# train
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m transformer.train --params exp_params/params_6.json --batch_size 512 --n_worker 56
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m transformer.train --params exp_params/params_7.json --batch_size 512 --n_worker 56
