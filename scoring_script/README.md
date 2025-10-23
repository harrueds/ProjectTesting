# Credit Scoring with Deep Neural Networks

## 1. Overview

This project develops a credit scoring model using deep neural networks (DNNs) to estimate the probability of customer default.
It pretends to be a script version of the notebook `scoring_crediticio_RNP.ipynb` presents in the repository [Scoring_Crediticio_RNP](https://github.com/harrueds/Scoring_Crediticio_RNP)
The project focuses on the application of supervised machine learning techniques to real-world financial problems, considering the processing of structured data and the handling of class imbalances, common in credit scenarios.

The `credit_scoring.py` script (included in this repository) presents the complete process from data preparation to the final evaluation of the model, integrating both technical and interpretive aspects of deep learning.

---

## 2. Objectives

- Implement a reproducible workflow for a binary classification problem applied to credit risk.  
- Compare performance and stability metrics of the model using validation and ROC curves.  
- Analyze the influence of predictor variables using interpretability techniques (SHAP).  
- Promote good practices in data preprocessing, class balancing, and dense neural network design.  

---

## 3. Technologies and Libraries Used

### **1. Data Manipulation**

- **NumPy (`numpy`)**: vectorized calculations and efficient matrix handling.  
- **Pandas (`pandas`)**: cleaning, exploration and tabular data transformation.

### **2. Visualization**

- **Matplotlib (`matplotlib.pyplot`)**: static graphs and learning curves.

### **3. Scikit-learn: Preprocessing and Evaluation**

- `train_test_split`: data division.  
- `OneHotEncoder`, `StandardScaler`: variable encoding and normalization.  
- `ColumnTransformer`: combination of heterogeneous transformations.  
- `compute_class_weight`: class weight adjustment for imbalanced data.  
- Metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `RocCurveDisplay`.

### **4. TensorFlow / Keras: Deep Learning Models**

- **TensorFlow (`tensorflow`)**: framework principal para deep learning.  
- **Keras (`tensorflow.keras`)**: construction of the model with:
  - `layers`: definition of dense architecture.  
  - `regularizers`: control of overfitting (L1, L2).  
  - `callbacks`: training strategies (`EarlyStopping`, `ReduceLROnPlateau`).  
  - `Model`: creation and compilation of the custom model.

### **5. Interpretability**

- **SHAP (`shap`)**: interpretation of the contribution of each variable to the final model result.

### **6. Others**

- **Warnings**: controlled silencing to improve the readability of the results.

---

## 4. Notebook Structure

1. **Data loading and exploration**  
   - Review of null values, variable types and class distribution.  
2. **Data preprocessing**  
   - Categorical variable encoding and numeric variable standardization.  
   - Data division into training and test sets.  
3. **Model construction**  
   - Sequential dense architecture with regularization and ReLU / Sigmoid activation functions.  
   - Optimization with *Adam* and monitoring through *EarlyStopping*.  
4. **Model evaluation**  
   - Calculation of performance metrics and ROC curve visualization.  
   - Confusion matrix generation.  
5. **Interpretability**  
   - Analysis of SHAP values to understand the contribution of variables to the final model result.  
6. **Conclusion**  
   - Discussion on performance, bias, stability and generalization capacity of the model.

---

## 5. Main Results

The deep learning model achieved satisfactory metrics of **precision and sensitivity**, showing an appropriate capacity to discriminate between customers with and without the risk of non-payment.  
The use of **balanced class weights** allowed mitigating bias towards the majority class, while regularization L2 and early stopping prevented overfitting.

The analysis with **SHAP** revealed the most influential variables in the prediction, reinforcing the interpretability of the model, an aspect crucial in financial applications.

---

## 6. Personal Reflection

During the development of this project, I learned to design a complete machine learning workflow oriented to the financial sector, understanding the particularities of imbalanced problems and the need for transparency in predictive models.  
The use of **TensorFlow/Keras** allowed me to master the creation of dense architectures and hyperparameter configuration, while the use of **SHAP** reinforced the importance of interpretability in sensitive contexts such as credit.

---

## 7. Execution in Local Environment

### 🔹 Previous Requirements

- Python **3.10+**
- pip and venv (for virtual environments) or `conda` (recommended)

### 🔹 Environment Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment (Linux / macOS)
source venv/bin/activate

# Activate environment (Windows)
venv\Scripts\activate
```

### 🔹 Recommended use with conda (recommended for reproducibility)

If you use conda, it is preferred to create a specific environment for the project and use `pip` only for packages not available or recommended by pip (for example TensorFlow). I include an `environment.yml` file in the repo to create the reproducible environment:

```bash
# create the environment from environment.yml
conda env create -f environment.yml
conda activate rnp

# después, si necesitas una versión específica de TensorFlow o GPU
# es común instalar TensorFlow por pip dentro del entorno conda:
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'tensorflow>=2.10' shap
```

Notes:
- The `environment.yml` uses `conda-forge` for most binary packages.
- It is recommended to use TensorFlow 2.x (>=2.10 in `environment.yml`) for compatibility with `tf.keras`.
- If you work with GPU, follow the official TensorFlow instructions to install compatible versions of CUDA/cuDNN.

### 🔹 VS Code

If you use VS Code, make sure to select the Python interpreter from the conda environment created (Command Palette → Python: Select Interpreter). Otherwise, the analyzer (pylance) may show import errors even though the code runs correctly in the correct environment.


### 🔹 Optimizer and type checker compatibility

In the project code (`scoring_credito.py`), the optimizer of the model is passed as a name (string) —for example `"adam"`— in the call to `model.compile(optimizer=...)`. This was done deliberately to avoid warnings from the static analyzer (pylance/mypy) that, in some configurations, expect `optimizer: str` in the signature of `compile`.

In general, Keras accepts both styles:
- Pass the string with the optimizer name (recommended for static compatibility):

```python
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

- Pass the optimizer instance (more explicit and configurable):

```python
from tensorflow.keras import optimizers
opt = optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])  # type: ignore[arg-type]
```

If you prefer the second option and want to avoid the static warning, add `# type: ignore[arg-type]` to the end of the line, as shown above. This does not affect execution, it only silences the type checking in the editor.

Common optimizer names accepted as strings: `"adam"`, `"rmsprop"`, `"sgd"`, `"adagrad"`, `"adadelta"`, `"nadam"`.

The default implementation of the project uses `"adam"`, which is equivalent to `optimizers.Adam()` with the default configuration.


### 🔹 Execution

Run the main script directly from the command line:

```bash
python3 credit_scoring.py
```

The script downloads the dataset, trains the models and generates the outputs (tables and figures) on the screen.

### 🔹 Author

- Henzo Alejandro Arrué Muñoz
- Developer with focus on Machine Learning and Linux

### 🔹 Contact

- Email: [harrueds@gmail.com](mailto:harrueds@gmail.com)

- GitHub: [harrueds](https://github.com/harrueds)
