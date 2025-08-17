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
from tensorflow.keras.utils import plot_model
from sklearn.tree import export_graphviz
from graphviz import Source
from graphviz import Digraph
import matplotlib.pyplot as plt
import tensorflow as tf


CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "scaler": StandardScaler(),
    "dnn_layers_units": [64, 32, 16],
    "dnn_activation": "relu",
    "dnn_optimizer": "adam",
    "dnn_epochs": 6,
    "dnn_batch_size": 32,
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
    # First hidden layer with input shape
    model.add(layers.Dense(config["dnn_layers_units"][0],
                           activation=config["dnn_activation"],
                           input_shape=(input_shape,)))

    # Remaining hidden layers
    for units in config["dnn_layers_units"][1:]:
        model.add(layers.Dense(units, activation=config["dnn_activation"]))
    model.add(layers.Dense(units, activation=config["dnn_activation"]))
    model.add(layers.Dropout(0.3))
    # Output layer (binary classification)
    model.add(layers.Dense(1, activation=config["dnn_activation"]))

    # Compile model
    model.compile(
        optimizer=config["dnn_optimizer"],
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate_dnn(X_train, y_train, X_val, y_val, X_test, y_test, config=CONFIG):
    model = create_dnn_model(X_train.shape[1], config=config)

    # Training
    history = model.fit(
        X_train, y_train,
        epochs=config["dnn_epochs"],
        batch_size=config["dnn_batch_size"],
        validation_data=(X_val, y_val),
        verbose=config["dnn_verbose"]
    )

    # Evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ DNN Test Accuracy: {test_accuracy:.2f}")

    # Predictions & classification report
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\n📊 DNN Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Optional: visualize if you have your utility ready
    try:
        visualize_dnn_with_graphviz(model, filename="./experiment/dnn_model_graphviz")
    except NameError:
        print("⚠️ Skipping Graphviz visualization (function not defined).")

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

        # Visualization for Random Forest
        if name == "Random Forest" and hasattr(model, "estimators_"):
            from sklearn.tree import export_graphviz
            from graphviz import Source

            tree = model.estimators_[0]  # Visualize the first tree
            dot_data = export_graphviz(
                tree,
                out_file=None,
                feature_names=[f"Feature {i}" for i in range(X_train.shape[1])],
                class_names=["Class 0", "Class 1"],
                filled=True,
                rounded=True,
                special_characters=True
            )
            graph = Source(dot_data)
            graph.render("./experiment/random_forest_tree", format="png", cleanup=True)
            print("Random Forest tree visualization saved as 'random_forest_tree.png'.")

        # Conceptual visualization for SVM and Logistic Regression
        if name in ["SVM", "Logistic Regression"]:
            from graphviz import Source

            dot_data = f"""
            digraph G {{
                rankdir=LR;
                node [shape=box, style=filled, color=lightblue];
                Input -> "{name} Model" -> Output;
            }}
            """
            graph = Source(dot_data)
            graph.render(f"experiment/{name.lower().replace(' ', '_')}_model", format="png", cleanup=True)
            print(f"{name} conceptual visualization saved as '{name.lower().replace(' ', '_')}_model.png'.")

    return trained_models

def visualize_dnn_with_graphviz(model, filename="dnn_model_graphviz"):
    """Visualize the DNN model architecture using graphviz."""
    from graphviz import Digraph

    graph = Digraph(format="png", name=filename)
    graph.attr(rankdir="LR", size="8,5")

    # Add input layer
    graph.node("Input", shape="circle", style="filled", color="lightblue", label="Input Layer")

    # Add hidden layers
    for i, units in enumerate(CONFIG["dnn_layers_units"]):
        graph.node(f"Hidden_{i+1}", shape="circle", style="filled", color="lightgreen", label=f"Hidden Layer {i+1}\n({units} units)")
        if i == 0:
            graph.edge("Input", f"Hidden_{i+1}")
        else:
            graph.edge(f"Hidden_{i}", f"Hidden_{i+1}")

    # Add output layer
    graph.node("Output", shape="circle", style="filled", color="lightcoral", label="Output Layer\n(1 unit)")
    graph.edge(f"Hidden_{len(CONFIG['dnn_layers_units'])}", "Output")

    # Render the graph
    graph.render(filename, cleanup=True)
    print(f"DNN model architecture saved as '{filename}.png'")
    


# def explain_model_with_lime(model, X_train, X_test, feature_names, class_names, model_name):
#     """Explain a model's predictions using LIME and visualize with graphviz."""
#     explainer = LimeTabularExplainer(
#         training_data=X_train,
#         feature_names=feature_names,
#         class_names=class_names,
#         mode='classification',
#         discretize_continuous=True
#     )

#     # Select a random instance from the test set
#     instance_idx = np.random.randint(0, X_test.shape[0])
#     instance = X_test[instance_idx]

#     def predict_proba_fn(data):
#         if hasattr(model, 'predict_proba'):
#             return model.predict_proba(data)
#         else:
#             probs = model.predict(data)
#             if probs.shape[1] == 1:
#                 probs = np.hstack([1 - probs, probs])
#             return probs

#     try:
#         # Generate the explanation
#         explanation = explainer.explain_instance(
#             data_row=instance,
#             predict_fn=predict_proba_fn,
#             num_features=10
#         )

#         # Print the explanation in the notebook
#         print(f"\nLIME Explanation for {model_name}:\n")
#         explanation.show_in_notebook(show_table=True, show_all=False)

#         # Plot the explanation
#         fig = explanation.as_pyplot_figure()
#         plt.title(f"LIME Explanation for {model_name}")
#         plt.tight_layout()
#         plt.show()

#         # Create a graphviz visualization
#         graph = Digraph(format="png", name=f"{model_name}_lime_explanation")
#         graph.attr(rankdir="LR", size="8,5")

#         # Add the prediction node
#         graph.node("Prediction", shape="ellipse", style="filled", color="lightblue", label=f"Prediction: {class_names[np.argmax(predict_proba_fn([instance]))]}")

#         # Add feature contributions
#         for feature, weight in explanation.as_list():
#             color = "lightgreen" if weight > 0 else "lightcoral"
#             graph.node(feature, shape="box", style="filled", color=color, label=f"{feature}\n({weight:.2f})")
#             graph.edge(feature, "Prediction")

#         # Render the graph
#         graph.render(f"experiment/{model_name.lower().replace(' ', '_')}_lime_explanation", cleanup=True)
#         print(f"LIME explanation visualization saved as '{model_name.lower().replace(' ', '_')}_lime_explanation.png'.")

#     except Exception as e:
#         print(f"An error occurred during LIME explanation for {model_name}: {e}")
def explain_model_with_lime(model, X_train, X_test, feature_names, class_names, model_name, num_features=10):
    import numpy as np
    import lime
    import lime.lime_tabular
    from graphviz import Digraph
    import matplotlib.pyplot as plt

    # Ensure X_train is numpy array
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Pick a random instance from test set
    instance_idx = np.random.randint(0, X_test.shape[0])
    instance = X_test[instance_idx].reshape(-1)  # ensure 1D

    # Define predict_proba function compatible with sklearn & Keras models
    def predict_proba_fn(data):
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(data)
            if probs.shape[1] != len(class_names):
                # Ensure binary shape
                probs = np.hstack([1 - probs, probs])
        else:
            probs = model.predict(data)
            if probs.ndim == 1:
                probs = probs.reshape(-1, 1)
            if probs.shape[1] == 1:
                probs = np.hstack([1 - probs, probs])
        return probs

    try:
        # Initialize LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification",
            discretize_continuous=True,
            categorical_features=[]
        )

        # Generate explanation
        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_proba_fn,
            num_features=num_features
        )

        # Show explanation in notebook
        print(f"\nLIME Explanation for {model_name}:\n")
        explanation.show_in_notebook(show_table=True, show_all=False)

        # Plot explanation
        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation for {model_name}")
        plt.tight_layout()
        plt.show()

        # Create Graphviz visualization
        graph = Digraph(format="png", name=f"{model_name}_lime_explanation")
        graph.attr(rankdir="LR")
        graph.node("Instance", shape="ellipse", style="filled", color="lightgrey")

        # Add feature contributions
        for i, (feature, weight) in enumerate(explanation.as_list()):
            color = "lightgreen" if weight > 0 else "lightcoral"
            graph.node(f"f{i}", shape="box", style="filled", color=color, label=f"{feature}\n{weight:.3f}")
            graph.edge("Instance", f"f{i}")

        # Add prediction node
        pred_label = class_names[np.argmax(predict_proba_fn(instance.reshape(1, -1)))]
        graph.node("Prediction", shape="ellipse", style="filled", color="lightblue", label=f"Prediction: {pred_label}")
        for i in range(len(explanation.as_list())):
            graph.edge(f"f{i}", "Prediction")

        # Render graph
        graph.render(f"experiment/{model_name.lower().replace(' ', '_')}_lime_explanation", cleanup=True)
        print(f"LIME explanation visualization saved as 'experiment/{model_name.lower().replace(' ', '_')}_lime_explanation.png'")

    except Exception as e:
        print(f"An error occurred during LIME explanation for {model_name}: {e}")

def plot_training_curves(history, save_path=None):
    # Accuracy curves
    plt.figure(figsize=(12, 5))

    # ---- Accuracy ----
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("DNN Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # ---- Loss ----
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("DNN Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
from imblearn.over_sampling import SMOTE

def run_pipeline(df, config=CONFIG, balance=True):
    # Step 1: Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df, config)

    # Step 2: Balance classes if required
    if balance:
        smote = SMOTE(random_state=config.get("seed", 42))
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_val, y_val = smote.fit_resample(X_val, y_val)
        X_test, y_test = smote.fit_resample(X_test, y_test)

    # Step 3: Train and evaluate DNN
    dnn_model, dnn_history = train_and_evaluate_dnn(
        X_train, y_train, X_val, y_val, X_test, y_test, config
    )
    plot_training_curves(dnn_history)

    # Step 4: Train and evaluate ML models
    ml_models = train_and_evaluate_ml_models(X_train, y_train, X_test, y_test)

    # Step 5: Explain with LIME for DNN
    explain_model_with_lime(
        dnn_model,
        X_train,
        X_test,
        df.drop(columns=['class']).columns,
        ['Non-Seizure', 'Seizure'],
        "DNN"
    )

    # Step 6: Explain with LIME for each ML model
    for name, model in ml_models.items():
        explain_model_with_lime(
            model,
            X_train,
            X_test,
            df.drop(columns=['class']).columns,
            ['Non-Seizure', 'Seizure'],
            name
        )

    return {
        "dnn_model": dnn_model,
        "dnn_history": dnn_history,
        "ml_models": ml_models
    }
