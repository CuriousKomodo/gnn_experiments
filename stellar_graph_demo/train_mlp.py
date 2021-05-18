from sklearn import preprocessing
from stellargraph.datasets import datasets

from stellar_graph_demo.baseline.train_mlp_functions import create_train_val_test_datasets_mlp, get_mlp_model, \
    train_mlp_model, evaluate_mlp_model_on_test_dataset, visualise_mlp_embedding


if __name__ == "__main__":
    dataset = datasets.Cora()
    graph, node_subjects = dataset.load()

    print(graph.info())

    _features = graph.node_features()
    target_encoding = preprocessing.LabelBinarizer()
    _targets = target_encoding.fit_transform(node_subjects.values)

    _train_features, _val_features, _test_features, _train_targets, _val_targets, _test_targets \
        = create_train_val_test_datasets_mlp(
        features=_features,
        targets=_targets,
    )

    _model = get_mlp_model(
        input_size=_features.shape[1],
        num_labels=7
    )

    print(_model.summary())

    train_mlp_model(
        model=_model,
        train_features=_train_features,
        train_targets=_train_targets,
        val_features=_val_features,
        val_targets=_val_targets,
    )

    evaluate_mlp_model_on_test_dataset(
        model=_model,
        test_features=_test_features,
        test_targets=_test_targets,
    )

    visualise_mlp_embedding(
        model=_model,
        features=_features,
        targets=_targets,
        indices=node_subjects.index
    )
