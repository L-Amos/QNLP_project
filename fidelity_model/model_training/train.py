import sys
sys.path.append("../")  # Setting path so that imports can happen
import os
from lambeq import QuantumTrainer, SPSAOptimizer, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from fidelity_model import FidelityModel, fidelity_pqc_gen

def read_file(path, displayname=""):
    print(f"Reading {displayname}...", end="")
    csvfile = pd.read_csv(path)
    pairs = [[pair['sentence_1'], pair['sentence_2']] for i,pair in csvfile.iterrows()]
    labels = [pair['label'] for i,pair in csvfile.iterrows()]
    print("Done")
    return pairs, labels

def generate_circuits(pairs, description="Generating Circuits"):
    progress_bar = tqdm(pairs, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description(description)
    circuits = [fidelity_pqc_gen(sentence_1, sentence_2) for sentence_1, sentence_2 in progress_bar]
    return circuits

def create_dataset(circuits, labels, batch_size, displayname=""):
    print(f"Generating {displayname}...", end="")
    dataset = Dataset(circuits, labels, batch_size)
    print("Done")
    return dataset

def user_setup():
    param_start = float(input("Enter the start value for the training parameter.\n"))
    param_end = float(input("Enter the end value for the training parameter.\n"))
    param_step = float(input("Enter the step size between training parameters.\n"))
    epochs = int(input("Enter the number of training epochs\n"))
    batch_size = int(input("Enter the batch size\n"))
    return np.arange(param_start, param_end+param_step/2, param_step), epochs, batch_size

def training(model, train_dataset, val_dataset, param_vals, epochs, seed, c):
    print("TRAINING\n" + "="*len("TRAINING"))
    for a in param_vals:
        print(f"Learning Rate: {a}\n" + "-"*len(f"Learning Rate: {a}"))
        trainer = QuantumTrainer(
            model,
            loss_function=lambda x,y : np.mean((x-y)**2),  # MSE Loss
            epochs=epochs,
            optimizer=SPSAOptimizer,
            optim_hyperparams={'a': a, 'c': c, 'A':0.01*epochs},
            seed=seed
        )
        model.save('my_checkpoint.lt')  # Ensures we don't have to re-jit compile for every parameter
        train_costs = np.empty((5, epochs))
        val_costs = np.empty((5, epochs))
        for i in range(5):
            error = True
            while error:
                print(f"RUN {i+1}/5")
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
            train_costs[i] = trainer.train_epoch_costs[i*epochs:(i+1)*epochs]
            val_costs[i] = trainer.val_costs[i*epochs:(i+1)*epochs]
            print("")  # Separates the training outputs
        # Save averages
        path = f"data/{epochs}-EPOCHS"
        if not os.path.exists(path):
            os.mkdir(path)
        np.savetxt(f"{path}/a-{a}_train_costs.csv", np.mean(train_costs, axis=0), delimiter=',')
        np.savetxt(f"{path}/a-{a}_val_costs.csv", np.mean(val_costs, axis=0), delimiter=',')

def main():
    SEED = 2
    C = 0.06
    PARAMS, EPOCHS, BATCH_SIZE = user_setup()
    print("SETTING UP\n" + "="*len("SETTING UP"))
    train_pairs, train_labels = read_file("data/train_data.csv", "Train Data")
    val_pairs, val_labels = read_file("data/val_data.csv", "Val Data")
    train_circuits = generate_circuits(train_pairs, "Generating Train Circuits")
    val_circuits = generate_circuits(val_pairs, "Generating Val Circuits")
    train_dataset = create_dataset(train_circuits, train_labels, BATCH_SIZE, "Train Dataset")
    val_dataset = create_dataset(val_circuits, val_labels, BATCH_SIZE, "Val Dataset")
    print("Generating Model...", end="")
    model = FidelityModel.from_diagrams(train_circuits+val_circuits, use_jit=True)
    print("Done")
    training(model, train_dataset, val_dataset, PARAMS, EPOCHS, SEED, C)

main()
