import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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
    print(f"DNN Test Accuracy: {test_accuracy:.2f}")

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nDNN Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return model, history

def train_and_evaluate_ml_models(X_train, y_train, X_test, y_test):
    models = {
        "Random Forest": RandomForestClassifier(random_state=CONFIG["random_state"]),
        "SVM": SVC(probability=True, random_state=CONFIG["random_state"]),
        "Logistic Regression": LogisticRegression(random_state=CONFIG["random_state"])
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{name} Classification Report:\n")
        print(classification_report(y_test, y_pred))
        trained_models[name] = model

    return trained_models

def explain_model_with_lime(model, X_train, X_test, feature_names, class_names, model_name):
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True
    )

    instance_idx = np.random.randint(0, X_test.shape[0])
    instance = X_test[instance_idx]

    def predict_proba_fn(data):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(data)
        else:
            probs = model.predict(data)
            if probs.shape[1] == 1:
                probs = np.hstack([1 - probs, probs])
            return probs

    try:
        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_proba_fn,
            num_features=10
        )
        print(f"\nLIME Explanation for {model_name}:\n")
        explanation.show_in_notebook(show_table=True, show_all=False)
    except Exception as e:
        print(f"An error occurred during LIME explanation for {model_name}: {e}")
