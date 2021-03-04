#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=$DATA/finetuned-models/bert-random-1m-random/mlm.out
#SBATCH --error=$DATA/finetuned-models/bert-random-1m-random/mlm.err
#SBATCH --gres=gpu:v100:1

# reset modules
module purge

# load python module
module load python/anaconda3/2019.03

# activate the right conda environment
source activate $DATA/conda-envs/gab-language-change

# Useful job diagnostics
#
nvidia-smi
#

# Executing the finetuning script with set options
python run_finetuning.py \
    --model_name_or_path $DATA/adapted-models/bert-random-1m \
    --train_file $DATA/0_data/clean/labelled_ghc/train_random.csv \
    --validation_file $DATA/0_data/clean/labelled_ghc/eval_random.csv \
    --test_file $DATA/0_data/clean/labelled_ghc/eval_random.csv \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir $DATA/finetuned-models/bert-random-1m-random \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --max_seq_length 128 \
    --use_special_tokens