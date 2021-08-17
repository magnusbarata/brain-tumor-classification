# RSNA-MICCAI Brain Tumor Radiogenomic Classification

## Getting Started
### Installation
1. Clone this repository
   ```bash
   git clone https://github.com/magnusbarata/brain-tumor-classification.git
   cd brain-tumor-classification
   ```
2. Install the dependencies
   ```bash
   docker build -t tf-gpu setup
   docker run --gpus all -u $(id -u):$(id -g) -it --rm -v $(pwd):/work tf-gpu bash
   ```

### Train
1. Download the data and put it inside `data` directory.
2. Set the hyperparameter file and start training. 
   ```bash
   python train.py <EXP_DIR>
   ```
3. Monitor using tensorboard.
   ```bash
   tensorboard --logdir <EXP_DIR>/logs
   ```