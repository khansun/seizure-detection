import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import classification_report

CONFIG = {
    "test_size": 0.1,
    "random_state": 42,
    "scaler": StandardScaler(),
    "dnn_layers_units": [64, 32, 16],
    "dnn_activation": "relu",
    "dnn_optimizer": "adam",
    "dnn_epochs": 5,
    "dnn_batch_size": 16,
    "dnn_verbose": 1,
}

def prepare_data(df, config=CONFIG):
    if 'class' not in df.columns:
        raise ValueError("The dataset must contain a 'class' column.")

    X = df.drop(columns=['class']).values
    y = df['class'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config["test_size"], random_state=config["random_state"]
    )

    if config["scaler"]:
        X_train = config["scaler"].fit_transform(X_train)
        X_val = config["scaler"].transform(X_val)
        X_test = config["scaler"].transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dnn_model(input_shape, config=CONFIG):
    model = keras.Sequential()
    model.add(layers.Dense(config["dnn_layers_units"][0], activation=config["dnn_activation"], input_shape=(input_shape,)))

    for units in config["dnn_layers_units"][1:]:
        model.add(layers.Dense(units, activation=config["dnn_activation"]))

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=config["dnn_optimizer"], loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_dnn(X_train, y_train, X_val, y_val, X_test, y_test, config=CONFIG):
    model = create_dnn_model(X_train.shape[1], config=config)
    history = model.fit(
        X_train, y_train,
        epochs=config["dnn_epochs"],
        batch_size=config["dnn_batch_size"],
        validation_data=(X_val, y_val),
        verbose=config["dnn_verbose"]
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model, history



def explain_dnn_with_lime(dnn_model, X_train, X_test, feature_names, class_names):
    """
    Explains a prediction from a DNN model using LIME.

    Parameters:
        dnn_model: Trained DNN model.
        X_train: Training data used to initialize LIME.
        X_test: Test data to explain.
        feature_names: List of feature names.
        class_names: List of class names.

    Returns:
        None. Displays explanation in the notebook.
    """
    explainer = LimeTabularExplainer(
        training_data=X_train.values if hasattr(X_train, "values") else X_train,  # Ensure compatibility with pandas DataFrame
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True
    )

    instance_idx = np.random.randint(0, X_test.shape[0])
    instance = X_test.iloc[instance_idx] if hasattr(X_test, "iloc") else X_test[instance_idx]

    def predict_proba_fn(data):
        probs = dnn_model.predict(data)
        if probs.shape[1] == 1:  # If single-column probability output
            probs = np.hstack([1 - probs, probs])  # Convert to two-class probabilities
        return probs

    try:
        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_proba_fn,
            num_features=10
        )
        explanation.show_in_notebook(show_table=True, show_all=False)
    except Exception as e:
        print(f"An error occurred during LIME explanation: {e}")


