#!/bin/bash
#SBATCH --job-name="DRLAgentTraining"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=102GB
#SBATCH --time=01-00:00:00
#SBATCH --output="OUTPUT-%j.o"
#SBATCH --error="ERROR-%j.e"
#SBATCH --mail-type=END,FAIL
#SBATCH --account=csc790
#SBATCH --partition=small

# Load a software module
# module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8
# module load /deac/csc/classes/csc790/jianb21/Ensemble_stockTrading_2020/python/3.11.8

# Go into your work path
cd /deac/csc/classes/csc790/jianb21/Ensemble_stockTrading_2020/scripts/

# Run the program
python3 main.py