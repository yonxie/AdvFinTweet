#!/bin/bash

# echo '>>> downloading resource'
# wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BuvigAY_CgZfYIulFsQbJ9_fjWyKhmgC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BuvigAY_CgZfYIulFsQbJ9_fjWyKhmgC" -O resource.tar.gz && rm -rf cookies.txt
# tar -xvf resource.tar.gz

echo '>>> downloading dataset'
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1z_qBhY_Nvk-j4XxQP-Y8AmGEoJxXxCSS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1z_qBhY_Nvk-j4XxQP-Y8AmGEoJxXxCSS" -O data.tar.gz && rm -rf cookies.txt
tar -xvf data.tar.gz

# echo '>>> downloading checkpoints'
# wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qdllazs_NgSmA0DljyfyvLbms081Ti6x' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qdllazs_NgSmA0DljyfyvLbms081Ti6x" -O checkpoints.tar.gz && rm -rf cookies.txt
# tar -xvf checkpoints.tar.gz

# echo '>>> create folders'
# mkdir log
# mkdir log/train
# mkdir log/attack
