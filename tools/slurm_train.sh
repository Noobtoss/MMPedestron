#!/bin/bash
#
#SBATCH --job-name=MMPedestron
#SBATCH --output=R-%j.out
#SBATCH --error=E-%j.err
#SBATCH --mail-user=thomas.schmitt@th-nuernberg.de
#SBATCH --mail-type=ALL
#
#SBATCH --partition=p0
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

module purge
module load python/anaconda3
# module load cuda/cuda-11.4.4
eval "$(conda shell.bash hook)"
export PYTHONPATH=/nfs/scratch/staff/schmittth/sync/MMPedestron:$PYTHONPATH

conda activate MMPedestron-3.6

BASE_DIR=/nfs/scratch/staff/schmittth/sync/MMPedestron
CONFIG=$1
srun python -u tools/custom_dataset_test.py $BASE_DIR/$CONFIG
# srun python -u tools/train.py $BASE_DIR/$CONFIG --work-dir=$BASE_DIR/train_logs/$SLURM_JOB_ID --launcher='slurm' ${@:3}
