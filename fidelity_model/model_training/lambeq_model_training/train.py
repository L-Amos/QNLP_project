from os import sys
sys.path.append("../../")
from tqdm import tqdm
from lambeq import NumpyModel, BinaryCrossEntropyLoss, QuantumTrainer, SPSAOptimizer, Dataset
from utils import ingest, sentence_pqc_gen
import numpy as np

BATCH_SIZE = 30
EPOCHS = 120
SEED = 2

BCE = BinaryCrossEntropyLoss(use_jax=True)  # Loss Function

LANGUAGE_MODELS = {
    1: "DisCoCat",
    2: "BagOfWords",
    3: "WordSequence"
}

def main():
    LANGUAGE_MODEL = int(input("Which language model?\n1.\tDisCoCat\n2.\tBag of Words\n3.\tWord-Sequence\n"))
    # Ingest Data
    print("Ingesting...", end="")
    train_data = ingest("train_sentences.txt")
    train_sentences = train_data.keys()
    train_labels_unformatted = list(train_data.values())
    train_labels = [[int(label), 1-int(label)] for label in train_labels_unformatted]
    val_data = ingest("val_sentences.txt")
    val_sentences = val_data.keys()
    val_labels_unformatted = list(val_data.values())
    val_labels = [[int(label), 1-int(label)] for label in val_labels_unformatted]
    print("Done")
    # Generate train circuits
    progress_bar = tqdm(train_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description("Generating Test Sentence Circuits")
    train_circuits = [sentence_pqc_gen(sentence, LANGUAGE_MODEL) for sentence in progress_bar]
    # Generate val circuits
    progress_bar = tqdm(val_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description("Generating Val Sentence Circuits")
    val_circuits = [sentence_pqc_gen(sentence, LANGUAGE_MODEL) for sentence in progress_bar]
    # Load model + train
    print("Creating model...", end="")
    model = NumpyModel.from_diagrams(train_circuits+val_circuits, use_jit=True)
    print("Done")
    trainer = QuantumTrainer(
        model,
        loss_function=BCE,
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.2, 'c': 0.06, 'A':0.01*EPOCHS},
        seed=SEED
    )
    train_dataset = Dataset(
        train_circuits,
        train_labels,
        batch_size=BATCH_SIZE
    )
    val_dataset = Dataset(
        val_circuits,
        val_labels,
        batch_size=BATCH_SIZE
    )
    print("TRAINING")
    trainer.fit(train_dataset, val_dataset, log_interval=12)
    np.savetxt(f"data/{LANGUAGE_MODELS[LANGUAGE_MODEL]}_train_costs.csv", trainer.train_epoch_costs, delimiter=',')
    np.savetxt(f"data/{LANGUAGE_MODELS[LANGUAGE_MODEL]}_val_costs.csv", trainer.val_costs, delimiter=',')

main()
