import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from fidelity_model.model import FidelityModel
from fidelity_model.utils import fidelity_pqc_gen, ingest

def generate_circuits(pairs, description="Generating Circuits"):
    progress_bar = tqdm(pairs, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description(description)
    circuits = [fidelity_pqc_gen(sentence_1, sentence_2) for sentence_1, sentence_2 in progress_bar]
    return circuits

# Ingest data + generate PQCs
train_pairs, train_labels = ingest("experiments/loss_metric/data/train_data.csv", "Train Data")
val_pairs, val_labels = ingest("experiments/loss_metric/data/val_data.csv", "Val Data")
train_circuits = generate_circuits(train_pairs)
val_circuits = generate_circuits(val_pairs)

# Model Trained on MSE
model_mse = FidelityModel.from_checkpoint("experiments/loss_metric/runs/MSE/best_model.lt")
mse_fidelities = model_mse(train_circuits)
# Model Trained on MAE
model_mae = FidelityModel.from_checkpoint("experiments/loss_metric/runs/MAE/best_model.lt")
mae_fidelities = model_mae(train_circuits)

# Plot % below given threshold error for a range of thresholds
x = np.linspace(0, 1, 1000)
mse_errors = [np.count_nonzero(np.abs(mse_fidelities - train_labels) < threshold)/500*100 for threshold in x]
mae_errors = [np.count_nonzero(np.abs(mae_fidelities - train_labels) < threshold)/500*100 for threshold in x]

plt.plot(x, mse_errors, label="Trained with MSE")
plt.plot(x, mae_errors, label="Trained with MAE")
plt.xlabel("Threshold Error")
plt.ylabel("Percentage of Predictions Below Threshold Error")
plt.title(f"Comparison of Model Accuracy with Threshold Error")
plt.legend()
plt.grid()
plt.gca().invert_xaxis()
plt.show()
