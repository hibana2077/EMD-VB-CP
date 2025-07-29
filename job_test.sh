#!/bin/bash
#PBS -P rp06
#PBS -l ngpus=0
#PBS -l ncpus=16
#PBS -l mem=16GB
#PBS -l walltime=00:03:00
#PBS -l wd
#PBS -l storage=scratch/rp06

# 啟用 virtualenv
source /scratch/rp06/sl5952/EMD-VB-CP/.venv/bin/activate

# Run the experiment script
cd experiments
python demo.py