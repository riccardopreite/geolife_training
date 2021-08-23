#!/bin/bash
#SBATCH --job-name=user1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.azzalin@studio.unibo.it
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=user1
#SBATCH --gres=gpu:1

. venv/bin/activate # per attivare il virtual environment python

pip3 install -r requirements.txt

python3 -m 02_extract_stay_regions
