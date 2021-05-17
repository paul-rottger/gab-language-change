#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=19-cumu-pretrain
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXXX
#SBATCH --output=19-cumu-pretrain.out
#SBATCH --error=19-cumu-pretrain.err
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


# set year for which to adapt models (run separately and subsequently for subsequent years)
year="2019"

# set month to load initial model from
load_month="05"

for train_month in "06" "07" "08" "09" "10" "11" "12"; do
    
    echo "loading model from:" $load_month $year
    echo "adapting it to data from:" $train_month $year

    python run_mlm.py \
        --model_name_or_path $DATA/gab-language-change/adapted-models/reddit/cumu-models/bert-cumu_${year}_${load_month}_1m \
        --train_file $DATA/gab-language-change/0_data/clean/unlabelled_reddit/month_splits/train_${year}_${train_month}_1m.txt \
        --validation_file $DATA/gab-language-change/0_data/clean/unlabelled_reddit/total/test_rand_10k.txt \
        --save_steps 20000 \
        --use_special_tokens \
        --line_by_line \
        --do_train \
        --per_device_train_batch_size 64 \
        --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
        --output_dir $DATA/gab-language-change/adapted-models/reddit/cumu-models/bert-cumu_${year}_${train_month}_1m \
        --overwrite_output_dir \
        --num_train_epochs 1 \
        --max_seq_length 128

    # set load_month for next iteration to be adapt_month from this iteration
    load_month=$train_month
    
    # sleep just to make sure all the model saving from the adaptation script is done (not sure if necessary)
    sleep 15s

done

# 19/12 --> 20/01

python run_mlm.py \
    --model_name_or_path $DATA/gab-language-change/adapted-models/reddit/cumu-models/bert-cumu_2019_12_1m \
    --train_file $DATA/gab-language-change/0_data/clean/unlabelled_reddit/month_splits/train_2020_01_1m.txt \
    --validation_file $DATA/gab-language-change/0_data/clean/unlabelled_reddit/total/test_rand_10k.txt \
    --save_steps 20000 \
    --use_special_tokens \
    --line_by_line \
    --do_train \
    --per_device_train_batch_size 64 \
    --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
    --output_dir $DATA/gab-language-change/adapted-models/reddit/cumu-models/bert-cumu_2020_01_1m \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --max_seq_length 128


# 20/01 --> 20/02

python run_mlm.py \
    --model_name_or_path $DATA/gab-language-change/adapted-models/reddit/cumu-models/bert-cumu_2020_01_1m \
    --train_file $DATA/gab-language-change/0_data/clean/unlabelled_reddit/month_splits/train_2020_02_1m.txt \
    --validation_file $DATA/gab-language-change/0_data/clean/unlabelled_reddit/total/test_rand_10k.txt \
    --save_steps 20000 \
    --use_special_tokens \
    --line_by_line \
    --do_train \
    --per_device_train_batch_size 64 \
    --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
    --output_dir $DATA/gab-language-change/adapted-models/reddit/cumu-models/bert-cumu_2020_02_1m \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --max_seq_length 128
