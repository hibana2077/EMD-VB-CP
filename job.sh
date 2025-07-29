#!/bin/bash
#PBS -P rp06
#PBS -l ngpus=0
#PBS -l ncpus=32
#PBS -l mem=32GB
#PBS -l walltime=00:30:00
#PBS -l wd
#PBS -l storage=scratch/rp06

# 啟用 virtualenv
source /scratch/rp06/sl5952/EMD-VB-CP/.venv/bin/activate

# Run the experiment script
cd experiments
python run_experiments.py