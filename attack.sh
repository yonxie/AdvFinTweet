#!/bin/bash
echo ">>>>>>>>>>>>> Attack model $1 <<<<<<<<<<<<<<<<<<<"

prefix='formal'
steps=10

if [ $1 == 'han' ] 
then
    train_name='han_train_gsize100_hsize150_alpha0.1_nmsg30_nword40_lr0.1_wd1.0_schedule10_0.5' 
    ckpt_epoch=22
elif [ $1 == 'tweetgru' ]
then
    train_name='tweetgru_train_gsize100_hsize150_alpha0.1_nmsg30_nword40_lr0.01_wd1.0_schedule10_0.5' 
    ckpt_epoch=27
elif [ $1 == 'tweetlstm' ]
then
    train_name='tweetlstm_train_gsize100_hsize150_alpha0.1_nmsg30_nword40_lr0.01_wd1.0_schedule10_0.5' 
    ckpt_epoch=29
elif [ $1 == 'stocknet']
then
    train_name='stocknet_train_gsize100_hsize150_alpha0.1_nmsg30_nword40_lr0.01_wd1.0_schedule10_0.5' 
    ckpt_epoch=15
else
    echo 'Found unknown configuration!'
fi

python attack_main.py \
--paths_text 'text/tweet' \
--model_threshold  1 \
--pgd_prefix $prefix \
--pgd_model_name $1 \
--pgd_train_name $train_name \
--pgd_ckpt_epoch $ckpt_epoch \
--pgd_concat 1 \
--pgd_lr 0.5 \
--pgd_steps $steps \
--pgd_smooth 0 \
--pgd_m_num 5 \
--pgd_z_num 5 \
--pgd_r_num 1 \
--pgd_alt 0 \
--pgd_fix 0 \
--pgd_sparsity 1 \
--pgd_schedule_step 10 --pgd_schedule_gamma 0.1 \
--pgd_projection 0 \
--model_tech_feature_size 3 \
--pgd_optim 'adam' \
--pgd_attack_type 'replacement' \
--pgd_opt_method 'joint' \