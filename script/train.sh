#!/bin/bash

python make_dataset.py with server_user make_target=colored_mnist
python train.py with server_user colored_mnist skewed3 severity4
python train_vanilla.py with server_user colored_mnist skewed3 severity4