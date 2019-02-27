#!/bin/bash
#SBATCH -N 3
#SBATCH --job-name=fernanda_testing
#SBATCH -n 3
#SBATCH -c 1
#SBATCH --mem=50000
#SBATCH -o test_out.txt
#SBATCH -e test_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:2

module load anaconda/3.6
module load gnu7
module load openmpi3
module load cuda/10.0.130
module load matlab
source activate /opt/ohpc/pub/apps/pytorch_cuda10_mpi_gpudirect

python nn_HCP_v2.py
