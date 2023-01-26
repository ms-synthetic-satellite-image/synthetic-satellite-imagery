# Description
This is the main repository that keeps track of our working progress for the Satellite project. It contains codes, scripts, documents, etc, that are deliverables and helpers for the team.

## Generate Synthetic Satellite Imagery

We use a pretrained SPADE model to generate the synthetic satellite images, which will be used in the downstream experiments.

### Set-up

For now, please refer to the environment set up of [SPADE](https://github.com/nvlabs/spade/#installation)

### Steps

1. Go to `generate_synthetic` folder

2. Download model weights from this [folder](https://drive.google.com/drive/folders/11C1qxiOcIur7rWcom1odeCSQJ7g2sjmz) and place it in `./checkpoints/`

3. In terminal run `bash get_synthetic.sh [data_folder] [spade_model] [no_output] [gpu_id]` where `data_folder` is the path to maryland data folder, `spade_model` refers to name of model (`lambda_0`, `lambda_2`, ...), `no_output` is number of synthetic image generated, and `gpu_id` is the gpu to run the model on.


## Downstream Segmentation

### Set-up

For now, please install the environment using

```
conda env create -n ENVNAME --file environment.yml
```

### Steps

1. Go to `downstream_segmentation` folder

2. To train a model with a specified diversity value and mix rate:

```
python train.py --name [model name] --mix_rate [for example: 0.0, 0.5, 1.0, 2.0, 3.0] --lambda_diverse [for example: 0, 2, 4, 6, 8, 10]
```

3. To evaluate a trained downstream model:

```
python evaluation.py --model_path [path to a model checkpoint from step 2]
```