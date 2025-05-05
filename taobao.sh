#!/bin/bash

#SBATCH --job-name=taobao.out
#SBATCH --output=taobao.out
#SBATCH --error=taobao.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=seblar21@student.aau.dk

# popularity: True and False
# kernel: gaussian, poly, tanh 
# learning rate: 0.1, 0.05, 0.01
# sigma: only for gaussian kernel 0.5, 1.0, 2.0
# gamma: only for gaussian kernel 1.0, 2.0, 3.0
# embed_size: 32,64

# testing single values 
# DILAD_beta: 0.004 for TaoBao and 0.005 for Beauty
# k_list: 1,5,10,50,100,300
# DCC_alpha: 0.008
# FDCC_alpha: 0.008

singularity exec \
    --nv /ceph/project/dat803-25/dgl_25.03-py3.sif python "/ceph/project/dat803-25/DGRec/main.py" \
    --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC \ 
    --DILAD_beta 0.004