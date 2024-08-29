# Lander Game Neural Network

## Project Overview

This project implements a Feed-Forward Backpropagation Neural Network to solve a lander game problem. The objective is to safely land a spacecraft on a randomly generated landing zone without touching the unsafe terrain. The neural network is trained to control the lander by adjusting its thruster and turning, based on its position relative to the target.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Network Architecture](#network-architecture)
- [Training and Validation](#training-and-validation)
- [Testing and Performance](#testing-and-performance)
- [Usage](#usage)
- [Files](#files)
- [Acknowledgments](#acknowledgments)

## Data Collection

Data for training the neural network was collected by manually playing the lander game. Key details include:
- **Sample Size:** 3200 rows
- **Features:** 4 columns (all float type)
- **Data Processing:**
  - Duplicates were removed to improve model performance.
  - Missing values were filled using the average of the respective column.
  - Data was normalized to ensure all features are on a similar scale.

## Network Architecture

The neural network is a fully connected feed-forward network, consisting of multiple layers designed to map the lander's position to the appropriate thruster and turning actions. 

Key components:
- **Input Layer:** Two inputs representing the X and Y positions relative to the target.
- **Output Layer:** Two outputs controlling the lander thruster and turning.
- **Hidden Layers:** Customizable based on the complexity required.
- **Activation Functions:** Non-linear functions used in hidden layers to capture complex patterns in data.

## Training and Validation

- **Training Data:** 90% of the collected data.
- **Validation Data:** 10% of the collected data.
- **Epochs:** The network was trained for 100 epochs offline.
- **Hyperparameters:**
  - Learning Rate
  - Number of Hidden Neurons
  - Batch Size
- **Training Strategy:** The network was trained to balance the lander equally between left and right-side landings.

## Testing and Performance

The trained model was tested using the most recent weights. The performance was evaluated using the Root Mean Squared Error (RMSE) between the predicted and actual actions.

## Usage

To use the neural network, simply run the provided Python script. Ensure that the input data is formatted correctly and that the model weights are loaded before testing. 

Steps:
1. Clone the repository.
2. Prepare the data in the required format.
3. Run the training script to train the model.
4. Use the testing script to evaluate performance.

## Files

- `LL_Neural_Network`
- `NeuralNetHolder`
- `Neuron`
- `Pre_Processing`
- `weights_lunar_landar`
