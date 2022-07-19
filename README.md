
# A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction

We consider a tweet concatenation attack that 'retweet' semantically similar tweets to fool financial forecasting models. This repository is the official implementation of [our paper](https://openreview.net/pdf?id=Sxgh3cbSbq). 

<img src="https://github.com/yonxie/AdvFinTweet/blob/main/images/adversarial_tweet.png" width="400">


## Requirements

Run the following command to install requirements:

```setup
pip install -r requirements.txt
```

## Dataset & Trained Victim Models

The preprocessed dataset, used resources and trained victim models used in the paper are publicly available at: [dataset. resource and trained models](https://drive.google.com/drive/folders/1NX8eM7NlF9q-TGBDaV3YROGU_j6MTIJE?usp=sharing). 

Alternatively, run the following command to download and decompress the resources. It also creates the necessary folders used in the training and attacking. 

```
bash downloader.sh
```

## Attack & Evaluation

To run the attack and look into the results, run the following command with model arguements `han`, `stocknet`, `tweetgru` or `tweetlstm`:

```eval
bash attack.sh han
```

It conducts *concatenation attack with perturbation of replacement* for various budget via *joint optimization*. The results are saved in `/log/attack`. The attack uses our trained models. Change the arguments in the script to implement different attacks. 

## Training

To train the 4 victim models from the scratch, run this command with a model argument:

```train
bash train.sh han
```

It trains the models with the same hyperparameters used in the paper. The training logs and checkpoints are save in `/log/train` and `checkpoints` respectively. 

## Highlighted Results
- Effect of attack budget on Attack Success Rate (ASR)
<img src="https://github.com/yonxie/AdvFinTweet/blob/main/images/budget_effect.png" width="600">

- Impact of the attack on portfolio PnL: trading simulation with initial value `$10000` shows our attack causes additional loss of `$3200` (32%) over two years. 
<img src="https://github.com/yonxie/AdvFinTweet/blob/main/images/pnl.png" width="600">

## Citation

```
@article{xie2022advtweet, 
  title={A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction},
  author={Xie, Yong and Wang, Dakuo and Chen, Pin-Yu and Jinjun, Xiong and Liu, Sijia and Koyejo, Oluwasanmi},
  journal={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2022}
}
```
