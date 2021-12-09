# Intention Prediction

The objective is to predict the intention of the pedestrians in an image.

## Visualization

We use wandb for the visulizations. If you want to use features related to Weights & Biases (for experiment management/visualization), then you should create an associated account and run the following command.

```bash
wandb login
```

it will keep you login in the future.

## Setup

- Choose your parameters in the configs folder.

# Data

The input of the dataloader is a path to the processed data summary (csv file). The paths can be changed in the parameter file.
To get this file, you can use the following command.

- For the PIE dataset:
```bash
python data/prepare_data_pie.py --pie_path=<path_to_pie_data> --out_folder=<path_to_output_dir> --train_sets=<train_sets> --val_sets=<val_sets> --test_sets=<test_sets>
```
The train sets used are 0 1 2 3 4 and the val sets are 5 6

- For the JAAD dataset:
```bash
python data/prepare_data_jaad.py --jaad_path=<path_to_jaad_data> --out_folder=<path_to_output_dir> --train_ratio=<ratio> --val_ratio=<ratio> --test_ratio=<ratio>
```
The ratio used is 0.75, 0.25, 0.0.

## Training
```bash
CUDA_VISIBLE_DEVICES={GPU_ID} python main.py <config_path>
```
GPU_ID: The gpu id you want to use

## Sweeps

It is also possible to start a sweep (hyperparameter search) in wandb by running the following command:

```bash
wandb sweep confs/sweep.yaml
```

The configuration of the weep is stored in confs/sweep.yaml.
The sweep can also be directly started in your wandb experiment homepage.