#!/bin/bash
#SBATCH --clusters=tinygpu
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00

cd $HOME
# module load python/3.10-anaconda
source miniconda/bin/activate
srun python /home/kickelhack/scripts/train_pretrained.py 

dset:
  train: egs/debug/tr  # path to train folder, should contain both a noisy.json and clean.json file
  valid: egs/debug/tr  # path to the validation folder.
                       # If not set, the last epoch is kept rather than the best
  test: egs/debug/tr   # Path to the test set. Metrics like STOI and PESQ are evaluated on it
                       # every `eval_every` epochs.
  noisy_json: egs/debug/tr/noisy.json  # files to enhance. Those will be stored in the experiment
                                       # `samples` folder for easy subjective evaluation of the model.
  noisy_dir:
  matching: sort       # how to match noisy and clean files. For Valentini, use sort, for DNS, use dns.
eval_every: 2
