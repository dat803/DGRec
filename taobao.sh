#!/bin/bash

#SBATCH --job-name=taobao
#SBATCH --output=taobao_%j.out
#SBATCH --error=taobao_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=seblar21@student.aau.dk
#SBATCH --array=0-647%8    # TODO

datasets=("Beauty" "TaoBao")
embed_sizes=(32 64)
kernels=("gaussian" "poly" "tanh")
popularities=(False True)
learning_rates=(0.05 0.1 0.01)
sigmas=(1.0 0.5 2.0)
gammas=(2.0 1.0 3.0)

for dataset in "${datasets[@]}"; do
  for embed_size in "${embed_sizes[@]}"; do
    for kernel in "${kernels[@]}"; do
      for popularity in "${popularities[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
          for sigma in "${sigmas[@]}"; do
            for gamma in "${gammas[@]}"; do
              singularity exec --nv /ceph/project/dat803-25/dgl_25.03-py3.sif \
              python "/ceph/project/dat803-25/DGRec/main.py" \
                --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC \
                --dataset "$dataset" \
                --kernel "$kernel" \
                --embed_size "$embed_size" \
                --popularity "$popularity" \
                --lr "$learning_rate" \
                --sigma "$sigma" \
                --gamma "$gamma" \
                --DCC_alpha 0.008 \
                --FDCC_alpha 0.008 \
                --DILAD_beta $( [[ "$dataset" == "TaoBao" ]] && echo 0.004 || echo 0.005 ) \
                --k_list 1,5,10,50,100,300
            done
          done
        done
      done
    done
  done
done
