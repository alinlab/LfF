# Learning from Failure: Training Debiased Classifier from Biased Classifier

Official PyTorch implementation of ["Learning from Failure: Training Debiased Classifier from Biased Classifier"](https://arxiv.org/pdf/2007.02561.pdf) (NeurIPS 2020) by Junhyun Nam et al. 


## Install dependencies
```
pip install -r requirements.txt
```


## Make dataset
```
python make_dataset.py with server_user make_target=colored_mnist
```


## Training
```
python train.py with server_user colored_mnist skewed3 severity4
```

## Citation
```
@inproceedings{nam2020learning,
  title={Learning from Failure: Training Debiased Classifier from Biased Classifier},
  author={Junhyun Nam and Hyuntak Cha and Sungsoo Ahn and Jaeho Lee and Jinwoo Shin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
