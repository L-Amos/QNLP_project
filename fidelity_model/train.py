# %%
from lambeq import QuantumTrainer, SPSAOptimizer, Dataset
import numpy as np
import csv
from fidelity_model import FidelityModel, fidelity_pqc_gen

# Hyperparameters
EPOCHS=240
BATCH_SIZE=30
SEED=2
A=1
C=0.06

# Parse Train Data
print("Reading Train Data...")
with open("data/train_data.csv", "r", encoding="utf8") as f:
    csvfile = csv.DictReader(f)
    train_pairs = [[item['sentence_1'], item['sentence_2']] for item in csvfile]
    f.seek(0)  # Rewind to beginning
    train_labels = [item['label'] for item in csvfile][1:] # Slicing ignores header

print("Generating Train PQCs...")
train_circuits = [fidelity_pqc_gen(sentence_1, sentence_2) for sentence_1, sentence_2 in train_pairs]

# Parse Validation Data
print("Reading Val Data...")
with open("data/val_data.csv", "r", encoding="utf8") as f:
    csvfile = csv.DictReader(f)
    val_pairs = [[item['sentence_1'], item['sentence_2']] for item in csvfile]
    f.seek(0)  # Rewind to beginning
    val_labels = [item['label'] for item in csvfile][1:] # Slicing ignores header

print("Generating Val PQCs...")
val_circuits = [fidelity_pqc_gen(sentence_1, sentence_2) for sentence_1, sentence_2 in val_pairs]

def loss(predictions, labels):
    return np.mean((predictions-labels)**2)

model = FidelityModel.from_diagrams(train_circuits+val_circuits, use_jit=True)
trainer = QuantumTrainer(
    model,
    loss_function=loss,
    epochs=EPOCHS,
    optimizer=SPSAOptimizer,
    optim_hyperparams={'a': A, 'c': C, 'A':0.01*EPOCHS},
    seed=SEED
)

train_dataset = Dataset(
        train_circuits,
        train_labels,
        batch_size=BATCH_SIZE)
val_dataset = Dataset(
        val_circuits,
        val_labels,
        batch_size=BATCH_SIZE)

print("===TRAINING===")
trainer.fit(train_dataset, val_dataset, log_interval=1)

np.savetxt("latest_train_costs.csv", trainer.train_epoch_costs, delimiter=',')
np.savetxt("latest_val_costs.csv", trainer.val_costs, delimiter=',')