from sklearn import preprocessing
from stellargraph.datasets import datasets
from pprint import pprint
import numpy as np

from stellar_graph_demo.baseline.train_mlp_functions import (
    create_train_val_test_datasets_mlp,
    get_mlp_model,
    train_mlp_model,
    evaluate_mlp_model_on_test_dataset,
    visualise_mlp_embedding
)
from stellar_graph_demo.visualisation import tsne_plot_embedding


if __name__ == "__main__":
    dataset = datasets.Cora()
    graph, node_subjects = dataset.load()

    pprint(graph.info())

    # Obtain the node features from graph, which has dimension (2708, 1433)
    _features = graph.node_features()

    # Convert the string labels into one-hot vectors
    target_encoding = preprocessing.LabelBinarizer()
    _targets = target_encoding.fit_transform(node_subjects.values)

    # Visualise the initial embedding of all the nodes via TSNE
    gt_labels = np.argmax(_targets, axis=1)
    tsne_plot_embedding(
        X=_features,
        y=gt_labels,
        indices=node_subjects.index,
        model_name='MLP'
    )

    # Split the node features into train, validation and test
    _train_features, _val_features, _test_features, _train_targets, _val_targets, _test_targets \
        = create_train_val_test_datasets_mlp(
        features=_features,
        targets=_targets,
    )

    # Build baseline model: 2-layer MLP
    _model = get_mlp_model(
        input_size=_features.shape[1],
        num_labels=7
    )

    print(_model.summary())

    # Train model
    train_mlp_model(
        model=_model,
        train_features=_train_features,
        train_targets=_train_targets,
        val_features=_val_features,
        val_targets=_val_targets,
    )

    # Get classification accuracy of pre-trained model on the test node features
    evaluate_mlp_model_on_test_dataset(
        model=_model,
        test_features=_test_features,
        test_targets=_test_targets,
    )

    # Visualise the final embedding of all the nodes via TSNE - pre-softmax layer
    visualise_mlp_embedding(
        model=_model,
        features=_features,
        targets=_targets,
        indices=node_subjects.index
    )
