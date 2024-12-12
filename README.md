# AI Course Project: Image Classification with Neural Networks

## Overview
This repository contains the work for an AI course project focusing on image classification using various machine learning and deep learning techniques. The project investigates the performance of convolutional neural networks (CNNs) with ResNet18, a custom neural network for regression, and Random Forest for comparison.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [Setup Instructions](#setup-instructions)
- [Results and Insights](#results-and-insights)
- [Contributors](#contributors)

## Project Structure
```
├── ASE_Assignment-1.docx                     # Project report detailing objectives and methodologies
├── Image_CNN_ResNet18.ipynb                  # Notebook implementing CNN and ResNet18 for classification
├── CNN_regression.ipynb                      # Notebook implementing a neural network for regression
├── CNN_Landscape.ipynb                       # Notebook implementing a tenserflow nueral network
├── Image_CNN_resnet_charts.ipynb             # Notebook implemtning resnet model's accuracy charts
├── Image_CNN_resnet_graphs.ipynb             # Notebook implemtning resnet model's accuracy charts
├── Random_Forest_Image_Classification.ipynb  # Notebook implementing Random Forest for comparison
├── Random_Forest_Image_Classification.ipynb  # Notebook implemtning Random Forest model's accuracy charts
├── README.md                                 # Project description (this file)
```

## Dataset
- **Source**: [Dataset link or description, if applicable]
- **Description**: A collection of labeled images for classification.
- **Goal**: Classify images into predefined categories and compare model performance.

## Methodology
The project applies multiple methodologies to address the image classification problem:

1. **Convolutional Neural Networks (CNNs)**:
   - Implemented a ResNet18 model pre-trained on ImageNet.
   - Fine-tuned the model to adapt to the specific dataset.

2. **Regression with Neural Networks**:
   - Built a custom neural network to predict image-based regression outputs.

3. **Random Forest**:
   - Applied Random Forest as a benchmark for comparing performance against deep learning models.

4. **Evaluation Metrics**:
   - Accuracy for classification tasks.
   - Mean Squared Error (MSE) and R² score for regression tasks.

## Key Features
- **ResNet18 Model**:
  - Fine-tuned for image classification with transfer learning.
  - Adapted to small datasets for improved generalization.

- **Custom Regression Neural Network**:
  - Developed a fully connected neural network for regression tasks.

- **Random Forest Comparison**:
  - Applied Random Forest to evaluate its efficiency and limitations for image-based problems.

- **Visualization**:
  - Confusion matrices and learning curves to assess model performance.

## Setup Instructions
### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Libraries: PyTorch, scikit-learn, Matplotlib, Pandas, NumPy

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the repository:
   ```bash
   cd <repository_folder>
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Open and run the notebooks:
   ```bash
   jupyter notebook
   ```

## Results and Insights
- **Best Performing Model**: ResNet18 achieved the highest accuracy for image classification.
- **Regression Performance**:
  - Custom neural network provided reasonable predictions with an R² score of [value].
- **Random Forest Comparison**:
  - Highlighted the limitations of traditional machine learning models for complex image data.

## Contributors
- **Timur Mamadaliyev**: CNN implementation, regression models, Random Forest comparison, analysis, and reporting.
- [**Zhang Thai**](https://github.com/Jericho6688): CNN implementation, regression models
---
