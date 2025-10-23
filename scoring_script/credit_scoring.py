"""credit_scoring.py

Module to perform a complete credit scoring workflow using deep neural networks.
The purpose of the script is to allow the reproduction of training and evaluation
in environments without a graphical server (e.g. containers, remote nodes), saving
figures in disk.

Contents
--------
The module provides utilities for:

- download and prepare the German Credit dataset (UCI);
- build column transformers and class weights;
- build and compile two architectures (DNN and lightweight ResNet) with Keras;
- train models, evaluate metrics and generate graphics (ROC, SHAP);
- save results (CSV) and PNG figures in the script folder.

Notes
-----
- The backend non-interactive ``matplotlib.use('Agg')`` is used to avoid the
opening of graphic windows. For interpretation, the SHAP library is used;
the module dynamically detects the presence of ``shap.maskers``.

Author
------
Henzo Alejandro Arrué Muñoz
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import shap


# use non-interactive backend to avoid opening graphic windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)
import tensorflow as tf

# import símbolos Keras
import tensorflow.keras as tf_keras  # type: ignore

layers = tf_keras.layers
regularizers = tf_keras.regularizers
callbacks = tf_keras.callbacks
Model = tf_keras.Model
optimizers = tf_keras.optimizers
metrics = tf_keras.metrics

KERAS_BACKEND = "tf.keras"
import keras

layers = keras.layers
regularizers = keras.regularizers
callbacks = keras.callbacks
Model = keras.Model
optimizers = keras.optimizers
metrics = keras.metrics


def load_german_credit():
    """Download and load the German Credit dataset from UCI.

    Returns
    -------
    pandas.DataFrame
        DataFrame with original columns and a binary `target` column
        where 1 indicates non-payment and 0 indicates compliance.
    """

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    cols = [
        "status",
        "duration",
        "credit_history",
        "purpose",
        "amount",
        "savings",
        "employment_duration",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "present_residence",
        "property",
        "age",
        "other_installment_plans",
        "housing",
        "number_credits",
        "job",
        "people_liable",
        "telephone",
        "foreign_worker",
        "target",
    ]
    df = pd.read_csv(url, delim_whitespace=True, header=None, names=cols)
    df["target"] = (df["target"] == 2).astype(int)
    print(df.head())
    print(df.describe())
    print(df.shape)
    return df


def make_preprocessor(num, cat):
    """Build a `ColumnTransformer` that applies StandardScaler to numeric
    variables and OneHotEncoder to categorical variables.

    Parameters
    ----------
    num : list[str]
        List of numeric column names.
    cat : list[str]
        List of categorical column names.

    Returns
    -------
    ColumnTransformer
        ColumnTransformer object.
    """

    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat),
        ]
    )


def class_weights(y):
    """Calculate and return a dictionary of class weights for training.

    Parameters
    ----------
    y : array-like
        Training label vector.

    Returns
    -------
    dict
        Class weight mapping {class: weight}.
    """

    w = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return dict(zip(np.unique(y), w))


def compile_model(inp, out, name):
    """Compile a Keras model given input/output and name.

    Parameters
    ----------
    inp : keras.layers.Input
        Input layer.
    out : keras.layers.Layer
        Output layer.
    name : str
        Model name.

    Returns
    -------
    keras.Model
        Compiled model.
    """

    model = Model(inp, out, name=name)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[metrics.AUC(name="auc"), metrics.Precision(), metrics.Recall()],
    )
    return model


def build_dnn(input_dim, units=64, l2=1e-4, dr=0.2):
    """Build a simple DNN architecture.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    units : int
        Number of units per dense layer.
    l2 : float
        L2 regularization factor.
    dr : float
        Dropout rate.

    Returns
    -------
    keras.Model
        Compiled model.
    """

    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(units, "relu", kernel_regularizer=regularizers.l2(l2))(inp)
    x = layers.Dropout(dr)(x)
    x = layers.Dense(units, "relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dr)(x)
    return compile_model(inp, layers.Dense(1, "sigmoid")(x), "DNN")


def residual_block(x, units, l2, dr):
    """Build a simple residual block.

    Parameters
    ----------
    x : keras.Layer
        Input layer.
    units : int
        Unidades en las capas internas.
    l2 : float
        Factor de regularización L2.
    dr : float
        Dropout rate.

    Returns
    -------
    keras.Layer
        Output layer.
    """

    skip = x
    x = layers.Dense(units, "relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dr)(x)
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2))(x)
    return layers.Activation("relu")(layers.Add()([skip, x]))


def build_resnet(input_dim, units=64, blocks=3, l2=1e-4, dr=0.2):
    """Build a ResNet architecture with residual blocks.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    units : int
        Unidades base por bloque.
    blocks : int
        Number of residual blocks.

    Returns
    -------
    keras.Model
        Compiled model.
    """

    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(units, "relu", kernel_regularizer=regularizers.l2(l2))(inp)
    for _ in range(blocks):
        x = residual_block(x, units, l2, dr)
    return compile_model(inp, layers.Dense(1, "sigmoid")(x), "ResNet")


def get_callbacks():
    """Return the list of callbacks used in training.

    Returns
    -------
    list
        List of callbacks.
    """

    return [
        callbacks.EarlyStopping(
            "val_auc", mode="max", patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            "val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]


def evaluate(name, model, X, y, thresh=0.5):
    """Evaluate the model on X/y and return a dictionary with metrics.

    Parameters
    ----------
    name : str
        Model name.
    model : keras.Model
        Trained model.
    X : array-like
        Features.
    y : array-like
        True labels.
    thresh : float
        Threshold for converting probabilities to classes.

    Returns
    -------
    dict
        Dictionary with metrics and probabilities.
    """

    proba = model.predict(X, verbose=0).ravel()  # type: ignore[arg-type]
    y_pred = (proba >= thresh).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "cm": confusion_matrix(y, y_pred),
        "proba": proba,
    }


def run_shap(
    model, X_train, X_test, names, out_dir, max_display=15, bg_size=200, smp_size=200
):
    """Calculate SHAP explanations and save plots to `out_dir`.

    Parameters
    ----------
    model : keras.Model
        Model used for prediction (must have predict method).
    X_train, X_test : array-like
        Training and test sets (preprocessed).
    names : list[str]
        Feature names (for plots).
    out_dir : str
        Directory where to save images.
    max_display : int
        Maximum number of features to show in the bar plot.
    bg_size, smp_size : int
        Sample sizes for background and sample in SHAP.
    """

    rng = np.random.default_rng(42)

    bg = X_train[
        rng.choice(len(X_train), size=min(bg_size, len(X_train)), replace=False)
    ]
    sample = X_test[
        rng.choice(len(X_test), size=min(smp_size, len(X_test)), replace=False)
    ]

    def predict_fn(data):
        return model.predict(data, verbose=0)  # type: ignore[arg-type]

    maskers = getattr(shap, "maskers", None)
    if maskers is not None and hasattr(maskers, "Independent"):
        explainer = shap.Explainer(predict_fn, masker=maskers.Independent(bg))
    else:
        explainer = shap.Explainer(predict_fn)
    vals = explainer(sample)

    # save plots
    bar_path = os.path.join(out_dir, "shap_bar.png")
    summary_path = os.path.join(out_dir, "shap_summary.png")
    shap.plots.bar(vals, max_display=max_display, show=False)
    plt.savefig(bar_path, bbox_inches="tight")
    plt.clf()
    try:
        shap.summary_plot(vals.values, sample, feature_names=names, show=False)
        plt.savefig(summary_path, bbox_inches="tight")
        plt.clf()
    except Exception:
        # some shap versions produce interactive HTML; ignore
        pass


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    out_dir = base_dir

    df = load_german_credit()
    num = [
        "duration",
        "amount",
        "installment_rate",
        "present_residence",
        "age",
        "number_credits",
        "people_liable",
    ]
    cat = [c for c in df.columns if c not in num + ["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns="target"),
        df["target"],
        test_size=0.2,
        stratify=df["target"],
        random_state=42,
    )

    pre = make_preprocessor(num, cat)
    X_train_t, X_test_t = pre.fit_transform(X_train), pre.transform(X_test)
    names = pre.get_feature_names_out()
    cw = class_weights(y_train)

    dnn = build_dnn(X_train_t.shape[1])
    resnet = build_resnet(X_train_t.shape[1])
    cbs = get_callbacks()

    dnn.fit(
        X_train_t,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        class_weight=cw,
        callbacks=cbs,
        verbose=0,  # type: ignore[arg-type]
    )
    resnet.fit(
        X_train_t,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        class_weight=cw,
        callbacks=cbs,
        verbose=0,  # type: ignore[arg-type]
    )

    res_dnn, res_res = evaluate("DNN", dnn, X_test_t, y_test), evaluate(
        "ResNet", resnet, X_test_t, y_test
    )

    # save results table
    out_df = (
        pd.DataFrame([res_dnn, res_res])
        .drop(columns=["cm", "proba"])
        .set_index("model")
        .round(4)
    )
    out_df.to_csv(os.path.join(out_dir, "results.csv"))

    # ROC plot
    fig, ax = plt.subplots(figsize=(10, 8))
    RocCurveDisplay.from_predictions(y_test, res_dnn["proba"], name="DNN", ax=ax)
    RocCurveDisplay.from_predictions(y_test, res_res["proba"], name="ResNet", ax=ax)
    ax.legend()
    roc_path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.clf()

    #  SHAP
    run_shap(resnet, X_train_t, X_test_t, names, out_dir)
