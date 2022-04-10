#!/bin/bash
echo ">>>>>>>>>>>>> Train model $1 from scratch <<<<<<<<<<<<<<<<<<<"
python train_main.py \
--paths_text 'text/tweet' \
--paths_technical 'technical' \
--model_threshold  1 \
--train_epochs 100 \
--model_prefix 'formal' \
--train_lr 0.01 \
--train_weight_decay 0.1 \
--train_schedule_step 10 \
--train_schedule_gamma 0.5 \
--train_resume 0 \
--train_epochs 1 \
--model_tech_feature_size 3 \
--model_name $1 \
--model_alpha 0.1 \
--model_kl_lambda_aneal_rate 0.0001 \