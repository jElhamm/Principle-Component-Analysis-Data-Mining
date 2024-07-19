# Principal Component Analysis (PCA) Data Mining

   This project demonstrates the implementation of PCA using various popular libraries such as [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [PyTorch](https://pytorch.org/), [Scikit-learn](https://scikit-learn.org/), and [TensorFlow](https://www.tensorflow.org/). Each implementation is contained within its own [Jupyter Notebook](https://jupyter.org/), providing a comprehensive and detailed guide on how to perform PCA using these different tools.


## Repository Structure

   - [Implementation of PCA using SciPy](PCA_Implement_With_SciPy.ipynb)
   - [Implementation of PCA using PyTorch](PCA_Implement_With_PyTorch.ipynb)
   - [Implementation of PCA using Scikit-learn](PCA_Implement_With_Scikitlearn.ipynb)
   - [Implementation of PCA using TensorFlow](PCA_Implement_With_Tensorflow.ipynb)

## Dataset

   The dataset used in this project is the [`heart_statlog_cleveland_hungary_final.csv`](heart_statlog_cleveland_hungary_final.csv), which contains various features related to heart disease.
   This dataset serves as a benchmark for evaluating dimensionality reduction techniques like PCA.

## Requirements

   To run these notebooks, you will need the following libraries installed in your Python environment:

   | Library        | Version     | Implementation                  |
   |----------------|-------------|---------------------------------|
   | NumPy          | >= 1.21.0   | All implementations             |
   | Pandas         | >= 1.3.0    | All implementations             |
   | Matplotlib     | >= 3.4.2    | All implementations             |
   | Scikit-learn   | >= 1.0.0    | PCA_Implement_With_Scikitlearn   |
   | PyTorch        | >= 1.9.0    | PCA_Implement_With_PyTorch      |
   | TensorFlow     | >= 2.5.0    | PCA_Implement_With_Tensorflow   |

   You can install these dependencies using pip:

```bash
   pip install -r requirements.txt
```

## Overview

   * Implement With PyTorch

   This code details the process of implementing PCA from scratch using PyTorch.


      It covers the following steps:
      - Data preprocessing
      - Computing covariance matrices
      - Performing eigenvalue decomposition
      - Selecting principal components
      - Transforming the dataset

   * Implement With SciPy

   This code shows how to leverage SciPy's linear algebra capabilities for PCA.


      It covers the following steps:
      - Using SciPy for matrix operations
      - Simplifying eigenvalue decomposition with SciPy functions
      - Comparing results with other implementations

   * Implement With Scikit-learn

   This notebook demonstrates PCA using Scikit-learn, which provides a straightforward implementation.


      It covers the following steps:
      - Using Scikit-learn’s PCA class
      - Analyzing explained variance
      - Visualizing principal components

   * Implement With TensorFlow

   This code demonstrates PCA using TensorFlow.


      It covers the following steps:
      - Utilizing TensorFlow for tensor operations
      - Implementing PCA with TensorFlow's high-level functions
      - Comparing performance with other implementations

## Results and Analysis

   Each notebook concludes with a section on results and analysis, where we evaluate the performance of the PCA implementations on the heart disease dataset.
   We visualize the principal components and discuss the effectiveness of PCA in dimensionality reduction and data analysis.

## License

   This repository is licensed under the Apache License 2.0.
   See the [LICENSE](./LICENSE) file for more details.