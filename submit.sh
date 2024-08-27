#!/bin/bash

#SBATCH --account=def-adml2021
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000MB
#SBATCH --gres=gpu:1

module load python/3.11.4

source ../.venv/bin/activate

python alliance.py