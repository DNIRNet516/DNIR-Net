# DNIR-Net: Dual-Stage Restoration Method for Neutron Imaging with Industrial Benchmark Dataset

[Paper](https://arxiv.org/abs/) 

<p align="center">
    <img src="assets/method_architecture.png">
</p>



# Installation
```
# clone DNIR-Net repository
git clone https://github.com/DNIRNet516/DNIR-Net.git
cd DNIR-Net

# create environment
conda create -n DNIR_Net python=3.10
conda activate DNIR_Net
pip install -r requirements.txt
```

# NeutronIND-Parts Dataset
数据集展示图片

Dounload: NeutronIND-Parts dataset can be downloaded from 链接.

# Pretrained Models
We provide the pretrained weights of the second-stage model (EdgeControlNet), as well as the SwinIR model we trained for degradation removal during Stage 2 training.During inference, the SwinIR model is employed as the first-stage restoration module, while the trained EdgeControlNet remains fixed across all tasks.

| Model Name | Description | HuggingFace | BaiduNetdisk | 
| :---------: | :----------: | :----------: | :----------: |
| EdgeControlNet | EdgeControlNet trained on NeutronIND-Parts Dataset | [download](https:) | N/A |
| SwinIR | SwinIR trained on NeutronIND-Parts Dataset | [download](https:) | N/A |

# Inference


# Train DNIR-Net
Stage 1:
Train a SwinIR model, which will be used for degradation removal during Stage 1 training.
```
```

Stage 2:
xxxxx一段话
```
```

# Acknowledgement
This project is based on DiffBIR and SwinIR. Thanks for their awesome work.

