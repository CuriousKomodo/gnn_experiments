from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


def tsne_plot_embedding(X, y, indices, model_name = 'GraphSAGE'):
    trans = TSNE(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=indices)
    emb_transformed["label"] = y

    alpha = 0.7

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        emb_transformed[0],
        emb_transformed[1],
        c=emb_transformed["label"].astype("category"),
        cmap="jet",
        alpha=alpha,
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    plt.title(
        f"TSNE visualization of {model_name} embeddings for cora dataset"
    )
    plt.show()
