# Predictive Maintenance: A Comparative Analysis of Data Sampling Techniques

This repository contains the code and resources for my thesis project, which focuses on enhancing machine failure prediction by addressing the challenge of imbalanced data. I explored and compared several data sampling techniques to determine their effectiveness in improving classification model performance.

The core of this work is a comparison between traditional methods like SMOTE and random undersampling against a more advanced Generative Adversarial Network (GAN) approach for creating synthetic minority class data.

## 📝 Project Objective

In industrial settings, machine failures are rare events. This leads to highly imbalanced datasets where the "failure" class is severely underrepresented. When trained on such data, machine learning models often become biased towards the majority class (non-failure) and perform poorly at predicting actual failures—the very events we want to prevent.

The goal of this project was to:
1.  Implement and evaluate different sampling methods to rebalance the dataset.
2.  Compare the performance of two classification models (Random Forest and SVM) trained on these rebalanced datasets.
3.  Assess the viability of using a GAN (specifically a WPGAN-GP via the TabGAN library) to generate high-quality synthetic data for the minority class.

## 📊 The Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**. It's a synthetic dataset that simulates real-world predictive maintenance scenarios and was created for this purpose.

-   **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
-   **Reference Paper:** [Explainable Artificial Intelligence for Predictive Maintenance Applications](https://ieeexplore.ieee.org/document/9268676) by Matzka, S.

The dataset contains 10,000 data points with features like air temperature, process temperature, rotational speed, torque, and tool wear. The target is to predict two types of failures: **Machine Failure** (a general flag) and specific failure modes. My analysis focuses on the binary classification of the `Machine Failure` target.

## 🛠️ Methodology

To tackle the class imbalance, I implemented four distinct sampling strategies before training the classifiers.

1.  **Baseline (Stratified Sampling):** The original, imbalanced data was split into training and testing sets while preserving the original percentage of samples for each class. This serves as our baseline for comparison.
2.  **Random Undersampling:** This method balances the dataset by randomly removing samples from the majority class. It's simple, but risks discarding potentially useful information.
3.  **SMOTE (Synthetic Minority Over-sampling Technique):** A popular oversampling method that creates new minority class instances by interpolating between existing ones.
4.  **WPGAN-GP (Wasserstein GAN with Gradient Penalty):** A sophisticated deep learning approach. I used the `TabGAN` library to train a Generative Adversarial Network on the minority class data. The GAN learns the underlying data distribution and generates new, realistic synthetic data points representing machine failures.

These rebalanced training datasets were then used to train a **Random Forest Classifier** and a **Support Vector Machine (SVM)**. Their performance was evaluated on a common, untouched test set.

## 🚀 How to Run This Project

To replicate the analysis, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-project-name.git](https://github.com/your-username/your-project-name.git)
    cd your-project-name
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Explore the analysis:**
    The complete workflow, from data preprocessing to model training and evaluation for each sampling method, is documented in the Jupyter Notebooks located in the `/notebooks` directory.

## 📂 Repository Structure

The project is organized to be clear and accessible:

```
├── .gitignore          # Specifies files for Git to ignore
├── README.md           # You are here!
├── requirements.txt    # Project dependencies
├── data/               # Contains the raw dataset
├── notebooks/          # Holds the Jupyter Notebooks with the analysis
└── references/         # Contains academic papers and documentation
```

## 📈 Key Findings

*(This is a placeholder for you to fill in with your specific results!)*

My analysis demonstrated that the choice of sampling technique has a significant impact on model performance. The WPGAN-GP method, in particular, showed promising results, often leading to a better balance between precision and recall for the minority class compared to traditional methods like SMOTE. This suggests that generating high-quality synthetic data is a powerful and viable strategy for tackling severe class imbalance in predictive maintenance applications.

For detailed metrics, confusion matrices, and visualizations, please see the final sections of the notebooks.
