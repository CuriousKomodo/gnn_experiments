import numpy as np
import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator, FullBatchNodeGenerator
from stellargraph.layer import GraphSAGE, GAT
from tensorflow.keras import layers, optimizers, losses, Model
from sklearn import preprocessing, model_selection

from stellar_graph_demo.visualisation import tsne_plot_embedding


def get_model_and_generator(model_name, graph, num_labels):
    if model_name == 'graphsage':
        batch_size = 50
        num_samples = [10, 5]
        generator = GraphSAGENodeGenerator(graph, batch_size, num_samples)

        gnn_model = GraphSAGE(
            layer_sizes=[32, 32],
            generator=generator,
            bias=True,
            dropout=0.5,
        )
    elif model_name == 'gat':
        generator = FullBatchNodeGenerator(graph, method="gat")

        gnn_model = GAT(
            layer_sizes=[8, 8],
            activations=["elu", "softmax"],
            attn_heads=8,
            generator=generator,
            in_dropout=0.5,
            attn_dropout=0.5,
            normalize=None,
        )
    else:
        raise Exception(f'Unknown model: {model_name}')

    x_inp, x_out = gnn_model.in_out_tensors()
    prediction = layers.Dense(units=num_labels, activation="softmax")(x_out)

    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )
    return model, generator


def create_gnn_generators_flows(node_subjects, generator):
    train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=0.1, test_size=None, stratify=node_subjects
    )
    val_subjects, test_subjects = model_selection.train_test_split(
        test_subjects, train_size=500, test_size=None, stratify=test_subjects
    )
    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_subjects)
    test_targets = target_encoding.transform(test_subjects)
    val_targets = target_encoding.transform(val_subjects)

    train_gen = generator.flow(train_subjects.index, train_targets)
    val_gen = generator.flow(val_subjects.index, val_targets)
    test_gen = generator.flow(test_subjects.index, test_targets)

    return train_gen, val_gen, test_gen


def train_gnn_model(model,
                    train_generator_flow,
                    val_generator_flow,
                    ):
    history = model.fit(
        train_generator_flow,
        epochs=50,
        validation_data=val_generator_flow,
        verbose=2,
        shuffle=False
    )
    sg.utils.plot_history(history)


def evaluate_gnn_model_on_test_dataset(model, test_generator_flow):
    test_metrics = model.evaluate(test_generator_flow)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))


def visualise_gnn_embedding(node_subjects, generator, model, model_name):
    all_nodes = node_subjects.index
    all_mapper = generator.flow(all_nodes)

    embedding_model = Model(inputs=model.input, outputs=model.get_layer('lambda').output)
    embbeding_matrix = embedding_model.predict(all_mapper)

    target_encoding = preprocessing.LabelBinarizer()
    gt_labels = np.argmax(target_encoding.transform(node_subjects), axis=1)

    tsne_plot_embedding(
        X=embbeding_matrix,
        y=gt_labels,
        indices=node_subjects.index,
        model_name=model_name,
    )
