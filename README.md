# DgCD: Discrepancy-guided Channel Dropout for Domain Generalization (submitted in CIKM'24)


We use DomainBed, the link below has details. 

https://github.com/facebookresearch/DomainBed

Here is the file we modified:

* domainbed/networks/resnet.py

Here is the file we added:

* domainbed/networks/DgCD.py


## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=./dataset
```

### Environments

Environment details used for the main experiments. Every experiment is conducted on a single NVIDIA GeForce RTX3090 GPU.

```
Environment:
	Python: 3.6.13
	PyTorch: 1.8.1+cu111
	Torchvision: 0.9.1+cu111
	CUDA: 11.1
	CUDNN: 8.1.0
	NumPy: 1.19.5
```

## How to Run

First, download PACS dataset in `dataset` folder. (You can find the download script in "/domainbed/script/download.py")

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py PACS --dataset PACS --data_dir ./dataset --algorithm ERM_DgCD
```


### Main experiments

Run command with hyperparameters (HPs):

```sh
data='PACS'
rho=0.05
ratio=0.33
for t in `seq 0 2`
do
    python train_all.py $data \
    --trial_seed $t \
    --data_dir ./dataset \
    --algorithm ERM_DgCD \
    --dataset $data \
    --lr 1e-5 \
    --resnet_dropout 0.0 \
    --weight_decay 1e-6 \
    --rho $rho \
    --ratio $ratio
done
```



## License

This project is released under the MIT license, included [here](./LICENSE).

This project include some code from [facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed) (MIT license), 
[khanrc/swad](https://github.com/khanrc/swad) (MIT license), and [kakaobrain/miro](https://github.com/kakaobrain/miro) (MIT license).
