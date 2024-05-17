import torch
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from pot import Pot
from potae import PoTAE
from utils.vector2shape import reverse_vector_polygon
from utils.prepare_dataset import get_finetune_dataset_mnist


def visualize_embeddings(embeddings, labels, save_name):
    """
    Cluster embeddings with labels in T-SNE method, then plot the visualization.
    Args:
        embeddings: a tensor or numpy array 
    """
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))

    # Plot each class separately
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(projections[indices, 0], projections[indices, 1], label=label, alpha=0.5)

    plt.title('t-SNE Visualization with Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)


def visualize_reconstruction(grounding, prediction, save_name, num_to_show=10):
    """
    Visualize reconstuction result, i.e., compare polygon shapes to check how well the polygon are reconstructed by the autoencoder
    Args:
        grounding: tensor or numpy array, original shapes vector
        prediction: the output of the autoencoder, with the sample format as grounding
    """
    ori_polygons = [reverse_vector_polygon(vec) for vec in grounding[:num_to_show]]
    new_polygons = [reverse_vector_polygon(vec) for vec in prediction[:num_to_show]]

    # Create a figure with subplots for each polygon
    fig, axs = plt.subplots(2, num_to_show, figsize=(5*num_to_show, 20))

    # Plot polygons from the first list
    for i, polygon in enumerate(ori_polygons):
        axs[0, i].set_title('Original polygon {}'.format(i+1))
        x, y = polygon.exterior.xy
        axs[0, i].plot(x, y)
        holes_x = [[coord[0] for coord in interior.coords] for interior in polygon.interiors]
        holes_y = [[coord[1] for coord in interior.coords] for interior in polygon.interiors]
        if holes_x:
            for hole_x, hole_y in zip(holes_x, holes_y):
                axs[0, i].plot(hole_x, hole_y)

    # Plot polygons from the second list
    for i, polygon in enumerate(new_polygons):
        axs[1, i].set_title('Reconstrcted Polygon {}'.format(i+1))
        x, y = polygon.exterior.xy
        axs[1, i].plot(x, y)
        holes_x = [[coord[0] for coord in interior.coords] for interior in polygon.interiors]
        holes_y = [[coord[1] for coord in interior.coords] for interior in polygon.interiors]
        if holes_x:
            for hole_x, hole_y in zip(holes_x, holes_y):
                axs[1, i].plot(hole_x, hole_y)

    # Show the plot
    plt.savefig(save_name)


def downstream_evaluate(model, test_data, test_labels):
    """
    Evaluate overall accuracy of shapes classification on downstream task
    Args:
        model: the pytorch model to evaluate
        test_data: a tensor of shape (batch_size, seq_len, fea_dim), the dataset for evaluation
        test_labels: a tensor of shape (batch_size, 1), the labels of corresponding shape
    """
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs, dim=-1)
        eval_acc = (predicted == test_labels).sum().item() / test_labels.size(0)
    return eval_acc


def main():
    # Loading evaluation dataset
    test_dataset = "./dataset/mnist_polygon_test_2k.npz"
    test_tokens, test_labels = get_finetune_dataset_mnist(file=test_dataset, train=False)

    # Evaluate pre-training performance
    pre_train_weights = "./weights/potae_pretrain_bs256_epoch100_runname-iconic-durian-30.pth"
    # pre_train_weights = "./weights/potae_pretrain_bs256_epoch100_runname-logical-bush-68.pth"
    pre_trained_model = PoTAE()
    pre_trained_model.load_state_dict(torch.load(pre_train_weights, map_location=torch.device('cpu')))

    pre_trained_model.eval()
    with torch.no_grad():
        hiddens, outputs, _ = pre_trained_model(test_tokens)
    visualize_reconstruction(test_tokens, outputs, save_name="./output/mnist_reconstructin_0.png", num_to_show=10)
    visualize_embeddings(hiddens, test_labels, save_name="./output/mnist_embedding_0.png")


    # # Evaluate shape classification (Downstream task)
    # fine_tune_model, lin_prob_model = Pot(), Pot()
    # fine_tune_weights = "./weights/fine_tune_pot.pth"
    # lin_prob_weights = "./weights/lin_prob_pot.pth"
    # fine_tune_model.load_state_dict(torch.load(fine_tune_weights))
    # lin_prob_model.load_state_dict(torch.load(lin_prob_weights))
    # fine_tune_acc = downstream_evaluate(fine_tune_model, test_tokens, test_labels)
    # lin_prob_acc = downstream_evaluate(lin_prob_model, test_tokens, test_labels)

    # print("fine_tune_acc: {}, lin_prob_acc: {}".format(fine_tune_acc, lin_prob_acc))
    # with open("./output/acc.txt", 'w') as f:
    #     f.write("fine_tune_acc: {}, lin_prob_acc: {}".format(fine_tune_acc, lin_prob_acc))
    # f.close()


if __name__ == "__main__":
    main()
