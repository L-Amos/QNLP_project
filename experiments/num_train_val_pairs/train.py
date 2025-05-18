from lambeq import SPSAOptimizer, Dataset
import numpy as np
from tqdm import tqdm
from fidelity_model.model import FidelityModel
from fidelity_model.utils import fidelity_pqc_gen, ingest
from fidelity_model.quantum_trainer import QuantumTrainer


LANGUAGE_MODELS = {
    1: "DisCoCat",
    2: "BagOfWords",
    3: "WordSequence"
}

SEED = 2
BATCH_SIZE=30
LANGUAGE_MODEL=1
A = 1
C = 0.06
EPOCHS = 240
RUNS = 5

def generate_circuits(pairs, language_model, description="Generating Circuits"):
    progress_bar = tqdm(pairs, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description(description)
    circuits = [fidelity_pqc_gen(sentence_1, sentence_2, language_model) for sentence_1, sentence_2 in progress_bar]
    return circuits

def create_dataset(circuits, labels, batch_size, displayname=""):
    print(f"Generating {displayname}...", end="")
    dataset = Dataset(circuits, labels, batch_size)
    print("Done")
    return dataset

def training(model, train_dataset, val_dataset):
    print("TRAINING\n" + "="*len("TRAINING"))
    trainer = QuantumTrainer(
        model,
        loss_function=lambda x,y : np.mean((x-y)**2),  # MSE Loss
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': A, 'c': C, 'A':0.01*EPOCHS},
        seed=SEED
    )
    model.save('my_checkpoint.lt')  # Ensures we don't have to re-jit compile for every parameter
    train_costs = np.empty((RUNS, EPOCHS))
    val_costs = np.empty((RUNS, EPOCHS))
    train_corr = np.empty((RUNS, EPOCHS))
    val_corr = np.empty((RUNS, EPOCHS))
    for i in range(RUNS):
        error = True
        while error:
            print(f"RUN {i+1}/{RUNS}")
            model.load('my_checkpoint.lt')
            # Parse Train Data
            try:
                trainer.fit(train_dataset, val_dataset, log_interval=12)
            except PermissionError as e:  # If there's an error with permissions, try training again
                with open("error.log", "a") as f:
                    f.write(str(e))
            else:
                error=False
        # Store costs
        train_costs[i] = trainer.train_epoch_costs[i*EPOCHS:(i+1)*EPOCHS]
        val_costs[i] = trainer.val_costs[i*EPOCHS:(i+1)*EPOCHS]
        train_corr[i] = trainer.train_corr[i*EPOCHS:(i+1)*EPOCHS]
        val_corr[i] = trainer.val_corr[i*EPOCHS:(i+1)*EPOCHS]
        print("")  # Separates the training outputs
    np.savetxt("results/train_costs_500-100.csv", np.mean(train_costs, axis=0), delimiter=',')
    np.savetxt("results/val_costs_500-100.csv", np.mean(val_costs, axis=0), delimiter=',')
    np.savetxt("results/train_corr_500-100.csv", np.mean(train_corr, axis=0), delimiter=',')
    np.savetxt("results/val_corr_500-100.csv", np.mean(val_corr, axis=0), delimiter=',')

def main():
    print("SETTING UP\n" + "="*len("SETTING UP"))
    train_pairs, train_labels = ingest("data/train_data_500.csv", "Train Data")
    val_pairs, val_labels = ingest("data/val_data_100.csv", "Val Data")
    train_circuits = generate_circuits(train_pairs, LANGUAGE_MODEL, "Generating Train Circuits")
    val_circuits = generate_circuits(val_pairs, LANGUAGE_MODEL, "Generating Val Circuits")
    train_dataset = create_dataset(train_circuits, train_labels, BATCH_SIZE, "Train Dataset")
    val_dataset = create_dataset(val_circuits, val_labels, BATCH_SIZE, "Val Dataset")
    print("Generating Model...", end="")
    model = FidelityModel.from_diagrams(train_circuits+val_circuits, use_jit=True)
    print("Done")
    training(model, train_dataset, val_dataset)

main()
