#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=/ceph/project/dat803-25/DGRec/slurm_logs/gridsearch_%A_%a.out
#SBATCH --error=/ceph/project/dat803-25/DGRec/slurm_logs/gridsearch_%A_%a.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=seblar21@student.aau.dk
#SBATCH --array=0-11%8

# Parse the config using Python
read gamma sigma learning_rate popularity kernel dataset <<< $(python3 -c "
import json
with open('configs.json') as f:
    config = json.load(f)[${SLURM_ARRAY_TASK_ID}]
print(config['gamma'], config['sigma'], config['learning_rate'], config['popularity'], config['kernel'], config['dataset'])
")

CMD="singularity exec --nv /ceph/project/dat803-25/dgl_25.03-py3.sif \
python /ceph/project/dat803-25/DGRec/main.py \
--metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC \
--dataset $dataset \
--kernel $kernel \
--popularity $popularity \
--lr $learning_rate \
--DCC_alpha 0.008 \
--FDCC_alpha 0.008 \
--DILAD_beta $( [[ \"$dataset\" == \"TaoBao\" ]] && echo 0.004 || echo 0.005 ) \
--k_list 10,20,100,300"

if [[ "$kernel" == "gaussian" ]]; then
  CMD="$CMD --sigma $sigma"
fi
if [[ "$kernel" == "tanh" || "$kernel" == "radial" || "$kernel" == "poly" ]]; then
  CMD="$CMD --gamma $gamma"
fi

eval $CMD
