#!/bin/bash
#SBATCH --job-name=blip_pretrain_4gpus     # create a short name for your job
#SBATCH -A c01949
#SBATCH -p gpu                   # partition to use
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core)
#SBATCH --nodes=2                # node count
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --ntasks-per-node=4      # total number of tasks per node, must match --gpus-per-node
#SBATCH --gpus-per-node=4        # number of gpus per node, max is 4.
#SBATCH --time=05:30:00          # total run time limit (HH:MM:SS)
#SBATCH --output=./logs-slurm/blip_pretrain_4gpus_outs.out
#SBATCH --error=./logs-slurm/blip_pretrain_4gpus_errors.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user-name=jkatama@iu.edu

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load python/gpu

python -m venv .venv
source .venv/bin/activate
pip install -user transformers


start=`date +%s.%N`
srun python pretraining.py --mode train \
                            --device cuda\
                            --epochs 10\
                            --batch_size 64\
                            --resume False\
                            --checkpoint_interval_mins 60\
                            --checkpoint_fraction 0.5\
                            --num_workers 8\


end=`date +%s.%N`
delta=`echo "$end - $start" | /usr/bin/bc`
echo "$Execution time: $delta"