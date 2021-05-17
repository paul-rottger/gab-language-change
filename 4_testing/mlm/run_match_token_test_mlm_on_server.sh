#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=100matchmlm-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXXX
#SBATCH --output=100matchmlm-test.out
#SBATCH --error=100matchmlm-test.err
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


for modelpath in $DATA/gab-language-change/adapted-models/reddit/month-models/bert-*/; do
    for testpath in $DATA/gab-language-change/0_data/clean/unlabelled_reddit/politics_test/test_*_5k.txt; do

        if [[ $(( 10#$(basename $modelpath | cut -c6-9) )) = $(( 10#$(basename $testpath | cut -c6-9) +0 )) ]] && \
        [[ $(( 10#$(basename $modelpath | cut -c11-12) +0 )) = $(( 10#$(basename $testpath | cut -c11-12) +0  )) ]]
        then

            echo $(basename $modelpath) $(basename $testpath)

            python token_test_mlm.py \
                --model_name_or_path $modelpath \
                --validation_file $testpath \
                --use_special_tokens \
                --line_by_line \
                --do_eval \
                --per_device_eval_batch_size 128 \
                --output_dir $DATA/gab-language-change/eval-results/mlm/reddit/token-test/max_seq_128 \
                --output_name $(basename $modelpath)-$(basename $testpath .txt) \
                --overwrite_output_dir \
                --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
                --max_seq_length 128

        fi

    done
done