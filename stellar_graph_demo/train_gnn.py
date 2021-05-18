from stellargraph import datasets

from stellar_graph_demo.gnn.train_gnn_functions import (
    get_model_and_generator,
    create_gnn_generators_flows,
    train_gnn_model,
    evaluate_gnn_model_on_test_dataset,
    visualise_gnn_embedding,
)


if __name__ == "__main__":
    _model_name = 'gat'

    dataset = datasets.Cora()
    _graph, _node_subjects = dataset.load()

    print(_graph.info())

    _model, _generator = get_model_and_generator(
        model_name=_model_name,
        graph=_graph,
        num_labels=7
    )

    print(_model.summary())

    train_gen, val_gen, test_gen = create_gnn_generators_flows(
        generator=_generator,
        node_subjects=_node_subjects
    )

    train_gnn_model(
        model=_model,
        train_generator_flow=train_gen,
        val_generator_flow=val_gen,
    )

    evaluate_gnn_model_on_test_dataset(
        model=_model,
        test_generator_flow=test_gen,
    )

    visualise_gnn_embedding(
        node_subjects=_node_subjects,
        generator=_generator,
        model=_model,
        model_name=_model_name,
    )
