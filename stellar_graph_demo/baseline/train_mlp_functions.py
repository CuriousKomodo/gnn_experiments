import numpy as np
import stellargraph as sg
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import Model
from sklearn.metrics import accuracy_score

from tensorflow.keras import losses
from sklearn import model_selection

from stellar_graph_demo.visualisation import tsne_plot_embedding


def create_train_val_test_datasets_mlp(features, targets):
    """
    Splits the dataset (features + targets) with stratification according to the following proportions :
    Train: 271, Validation: 500, Test: 1937
    """
    train_features, test_features, train_targets, test_targets = model_selection.train_test_split(
        features, targets, train_size=0.1, test_size=None, stratify=targets
    )
    val_features, test_features, val_targets, test_targets = model_selection.train_test_split(
        test_features, test_targets, train_size=500, test_size=None, stratify=test_targets
    )

    return train_features, val_features, test_features, train_targets, val_targets, test_targets


def get_mlp_model(input_size, num_labels):
    """Builds the baseline model - 2-layer MLP that takes the initial node features as input"""
    model = Sequential()
    model.add(Dense(32, input_dim=input_size, activation='relu', name='embedding_layer'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )
    return model


def train_mlp_model(model,
                    train_features,
                    train_targets,
                    val_features,
                    val_targets):
    """Trains the MLP model in batches"""
    history = model.fit(
        x=train_features,
        y=train_targets,
        epochs=200,
        batch_size=32,
        validation_data=(val_features, val_targets),
        verbose=2,
        shuffle=True,
    )
    sg.utils.plot_history(history)
    return model


def evaluate_mlp_model_on_test_dataset(model, test_features, test_targets):
    """Evaluate the pre-trained MLP model on test dataset"""
    test_predictions = model.predict(test_features)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    test_targets_labels = np.argmax(test_targets, axis=1)
    test_acc = accuracy_score(test_targets_labels, test_pred_labels)
    print(f"Test Set Accuracy: {test_acc}")


def visualise_mlp_embedding(model, features, targets, indices):
    """Visualises the first layer of MLP via TSNE, coloured by ground truth labels"""
    gt_labels = np.argmax(targets, axis=1)
    embedding_model = Model(
        inputs=model.input,
        outputs=model.get_layer('embedding_layer').output
    )
    embedding_matrix = embedding_model.predict(features)
    tsne_plot_embedding(
        X=embedding_matrix,
        y=gt_labels,
        indices=indices,
        model_name='MLP'
    )


def visualise_initial_embedding(features, targets, indices):
    """Visualises the first layer of MLP via TSNE, coloured by ground truth labels"""
    gt_labels = np.argmax(targets, axis=1)
    tsne_plot_embedding(
        X=features,
        y=gt_labels,
        indices=indices,
        model_name='MLP'
    )
