#!/bin/bash
python -m transformer.train -e experiments/exp105 -b 1024 --n_worker 56
python -m transformer.train -e experiments/exp103 -b 1024 --n_worker 56
python -m transformer.train -e experiments/exp104 -b 1024 --n_worker 56
