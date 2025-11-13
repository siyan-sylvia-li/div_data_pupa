#!/bin/bash
cd /ocean/projects/cis250134p/shared/div_data_pupa
module load anaconda3
conda activate pupa
python optimization_script.py --num_iters 3
python optimization_script.py --num_iters 3 --optimizer simba