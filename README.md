# Adaptive Structure-Aware Connectivity-Preserving Loss for Improved Road Segmentation in Remote Sensing Images

This project provides the framework for road semantic segmentation tasks in satellite or aerial imagery. It includes multiple deep learning models, a variety of advanced loss functions, and configurable training scripts.

## Features

- **Multiple Models**: Implements both a standard **U-Net** and a more complex **Attention-based Semantic Segmentation Network. SemSeg network**.
- **Advanced Loss Functions**: A comprehensive options of loss functions is available, including:
  - SAC Loss (the main contribution of this work)
  - Combinations like `BCE_Tversky`, `BCE_SAC_lcDice`, etc.
- **Flexible Training**: The training scripts are highly configurable via command-line arguments, allowing for easy experimentation with different models, loss functions, and hyperparameters.
- **Data Handling**: Includes scripts for data preparation and loading, designed to work with datasets like DeepGlobe.

## Project Structure

The project is organized into a modular structure to improve clarity and maintainability:

```
SAC_loss_road_segmentation/
├── models/
│   ├── SemSeg_Network.py
│   └── Unet_Network.py
├── pretrained_weights/
│   └── ... (contains .pth model weights)
├── src/
│   ├── data.py
│   ├── losses.py
│   ├── semseg_utils.py
│   ├── unet_train.py
│   └── unet_utils.py
├── semseg_train.py
├── test_interface.ipynb
├── requirements.txt
└── README.md
```

- **`semseg_train.py`**: The main script for training the Attention-based segmentation model.
- **`src/unet_train.py`**: The script for training the U-Net model.
- **`test_interface.ipynb`**: A Jupyter Notebook to easily test models on new images using pretrained weightsa nd visualize the results.
- **`models/`**: Contains the model architectures.
  - `SemSeg_Network.py`: Defines the attention-based semantic segmentation model.
  - `Unet_Network.py`: Defines the standard U-Net model architecture.
- **`pretrained_weights/`**: Contains pretrained model weights (`.pth` files) that can be loaded for inference.
- **`src/`**: Contains supporting code.
  - **`losses.py`**: Contains all the custom loss function implementations.
  - **`data.py`**: Handles data loading and preparation.
  - **`semseg_utils.py` / `unet_utils.py`**: Utility functions for training and data processing.
- **`requirements.txt`**: Lists the project dependencies.
- **`README.md`**: This file.

## Models

### U-Net (`models/Unet_Network.py`)

A standard U-Net architecture, which is a convolutional neural network designed for fast and precise image segmentation. It consists of an encoder path to capture context and a symmetric decoder path that enables precise localization.

### Attention-based Network (`models/SemSeg_Network.py`)

A more advanced segmentation network that incorporates attention mechanisms and dilated convolutions. This allows the model to focus on more relevant features and capture multi-scale context, potentially leading to better performance on complex scenes.

## Usage

### Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Training

You can train the models using their respective training scripts.

**Training the Attention-Based Model:**

To train the `SemSeg` model on the `deepglobe` dataset with the `BCE_SAC_lcDice` loss function for 150 epochs, you can run:

```bash
python semseg_train.py --dataset_name deepglobe --loss BCE_SAC_lcDice --epochs 150 --k1 0.7 --k2 0.3 --k3 0.5
```

**Training the U-Net Model:**

```bash
python src/unet_train.py
```

### Inference

The `test_interface.ipynb` notebook provides an easy-to-use interface for running inference on your own images. It allows you to:

1.  Load a pretrained model from the `pretrained_weights/` directory.
2.  Input a new image.
3.  Visualize the resulting segmentation mask.

To use it, simply open the notebook in a Jupyter environment and follow the instructions in the cells.

#### Command-Line Arguments for `semseg_train.py`:

- `--runname`: (Optional) A name for the training run (e.g., for logging with `wandb`).
- `--projectname`: (Optional) The project name for logging.
- `--dataset_name`: (Required) The name of the dataset to use (e.g., `deepglobe`).
- `--loss`: (Optional) The loss function to use. Options include `BCE_Tversky`, `BCE_simpSAC`, `BCE_SimpSAC_lcDice`, `lcdice`, `BCE_SAC_lcDice`.
- `--k1`, `--k2`, `--k3`: (Optional) Floating-point hyperparameters for certain loss functions.
- `--epochs`: The number of training epochs (default: 100).
- `--nottest`: A flag to disable a test mode if implemented.

## Citation

If you find this work useful, please consider citing the following paper:

@inproceedings{shojaei2025adaptive,

  title={Adaptive Structure-Aware Connectivity-Preserving Loss for Improved Road Segmentation in Remote Sensing Images},

  author={Shojaei, Sara and Bohl, Trevor and Palaniappan, Kannappan and Bunyak, Filiz},

  booktitle={Proceedings of the Winter Conference on Applications of Computer Vision},

  pages={1210--1218},
  
  year={2025}
}

