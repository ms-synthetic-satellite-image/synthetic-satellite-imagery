# Generate Synthetic Data

1. Download model weights from this [folder](https://drive.google.com/drive/folders/11C1qxiOcIur7rWcom1odeCSQJ7g2sjmz) and place it in `/checkpoints`

2. In terminal run `bash get_synthetic.sh [data_folder] [spade_model] [no_output] [gpu_id]` where `data_folder` is the path to maryland data folder, `spade_model` refers to name of model (`lambda_0`, `lambda_2`, ...), `no_output` is number of synthetic image generated, and `gpu_id` is the gpu to run the model on.
