#!/bin/bash -l
  
#SBATCH --job-name="story-template-extraction"
#SBATCH --account="s1090"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dylan.ashley@idsia.ch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu
module load PyExtensions

srun bash './tasks_'$SLURM_ARRAY_TASK_ID'.sh'
