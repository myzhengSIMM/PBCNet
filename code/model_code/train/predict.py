
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import numpy as np

@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    valid_prediction = []
    valid_labels = []
    valid_1_labels = []
    ref_2_labels = []
    rank = []
    file = []

    att__1 = []
    att__2 = []

    for batch_data in loader:
        graph1, graph2, pock, label, label1, label2, rank1, file_name = batch_data
        # to cuda
        graph1, graph2, pock, label, label1, label2 = graph1.to(device), graph2.to(device), pock.to(device), label.to(device), label1.to(
            device), label2.to(device)

        logits,_, att1,att2 = model(graph1,
                       graph2, pock)

        valid_prediction += logits.tolist()

        att__1 += att1.tolist()
        att__2 += att2.tolist()

        valid_labels += label.tolist()
        valid_1_labels += label1.tolist()
        ref_2_labels += label2.tolist()
        rank += rank1.tolist()
        file += file_name

    mae = mean_absolute_error(valid_labels, valid_prediction)
    rmse = mean_squared_error(valid_labels, valid_prediction) ** 0.5

    # ======== to 'kcal/mol' unit =======
    valid_labels_G = np.log(np.power(10, -np.array(valid_labels).astype(float)))*297*1.9872*1e-3
    valid_prediction_G = np.log(np.power(10, -np.array(valid_prediction).astype(float)))*297*1.9872*1e-3

    mae_g = mean_absolute_error(valid_labels_G, valid_prediction_G)
    rmse_g = mean_squared_error(valid_labels_G, valid_prediction_G) ** 0.5

    valid_prediction = np.array(valid_prediction).flatten()
    valid_prediction_G = np.array(valid_prediction_G).flatten()

    return mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, np.array(valid_labels), att__1, att__2