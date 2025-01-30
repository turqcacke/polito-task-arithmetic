# AML & DAAI 2024/2025 Project - Task Arithmetic Under Bias-Variance Trade-offs
Official repository for the "Task Arithmetic Under Bias-Variance Trade-offs" project - Advanced Machine Learning & Data Analysis and Artificial Intelligence Courses 2024/2025 @ PoliTo

## Getting Started
Make sure to have a CUDA capable device, supporting at least CUDA 11.8, installed and correctly configured on your system. 

(The base code of this project has been produced using CUDA 11.8 and Python 3.10.9)

Follow https://pytorch.org/get-started/locally/ to setup PyTorch (note that PyTorch comes pre-installed on Google Colab's environments, so you can skip this step)

Once you have properly setup everything, make sure you are in the correct directory and run from the command line:
```bash
pip install -r requirements.txt
```

### Dataset
Download to your disk/drive the datasets from the provided drive folder (see project report). Then, unzip them to some location.

At this point you should be able to run and edit the base code provided.

*NOTE: please, do not push the datasets into your github repository / your exam submission.*

## Base Code Structure
The starting code should already provide everything needed to easily extend it. Read carefully the specifications in the project report.

In the following, you can find a brief description of the included files.

| File/Folder | Description |
| ---- | ----------- |
| `args.py` | contains the function responsible for parsing each command line argument. |
| `datasets/` | contains the files with code to load data, build splits and dataloaders. |
| `utils.py` | contains several utilities to correctly setup the pipeline. |
| `task_vectors.py` | contains the code for building task vectors and/or load checkpoints. |
| `modeling.py` | contains the backbone architectures and modules used in the project. |
| `heads.py` | contains the logic to build and store the open-vocabulary classifiers used in the project. |
| `merged_model.py` | contains the logic to build model by task vectors. |
| `finetune.py` | contains the logic finetune pretrained model. |
| `eval_single_task.py` | contains the metrics evaluation logic. |
| `eval_task_addition.py` | contains logic for multi task metrics evaluation and finding best alpha. |





## Running The Experiments
#### Running with VsCode
To run with **VsCode**, go to `Run and Debug` section, choose script you want to run. Press `F5`.

#### Running with `bash`
To run the experiments you can use, copy and modify the provided launch script `launch_scripts/base.sh`, which should give you an idea on how to design the implementation of the missing files.
As an example, after producing the three missing files, you should be able to launch the experiments as
```
./launch_scripts/base.sh
```
Supported command args for script:
| Argument          | Description                              |
|------------------|------------------------------------------|
| `--data-location` | Path to the dataset                     |
| `--save`         | Directory for saving checkpoints        |
| `--batch-size`   | Batch size for training                 |
| `--lr`           | Learning rate                           |
| `--wd`           | Weight decay (regularization factor)    |
| `--balanced`     | Flag for data balancing                 |
| `--stop-criterion` | stopping criterion to use: `none` (default), `fim`, or `valacc`. |


*NOTE: you should upload with your code also the launch scripts as they are fundamental to reproduce your results.*

In the following, you can find a brief description of the relevant command line arguments when launching the experiments. They should suffice. However, if needed, you can add additional arguments by editing the `args.py` file.

### Basic Command Line Arguments
| Argument &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  | Description |
| -------- | ----------- |
| `--model` | the name of the architecture used in the experiments (keep it as `ViT-B-32`) |
| `--batch-size` | batch size used in the optimization procedure (default: `32`) |
| `--lr` | learning rate used in the optimization procedure (default: `1e-4`) |
| `--wd` | weight decay of the optimizer (default: `0.0`) |
| `--data-location` | path to the folder containing your unzipped dataset folders. |
| `--balance` | flag whether balance train dataset or not. |
| `--st-alpha` | alpha scaling (only for `prerained` and `finetuned` models). |
| `--st-model` | model to use for generating `eval_single_task.json` report. Posible options: [`pretrained`, `finetuned`, `merged`] |
| `--fisher` | flag whether calculate fiher for report or not. |
| `--stop-criterion` | Specifies the stopping criterion: `none` - Uses the final epoch, `fim` - Stops at the maximum FIM log-trace, `valacc` - Stops at the maximum validation accuracy. |


## Bug Reporting
You are encouraged to share in the project's dedicated Slack channel any bugs you discover in the code. It's incredibly valuable for everyone involved to be aware of any issues as soon as they arise, helping to address them promptly. Your contribution is greatly appreciated!
