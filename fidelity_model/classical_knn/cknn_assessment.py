from tqdm import tqdm
from os import sys
sys.path.append("../")
from fidelity_model import FidelityModel, fidelity_pqc_gen
from utils import ingest, sentence_pqc_gen, get_states
import numpy as np
from collections import Counter


def main():
    # Load Model
    LANGUAGE_MODEL = int(input("Which language model?\n1.\tDisCoCat\n2.\tBag of Words\n3.\tWord Sequence\n"))
    path = input("Enter path to model checkpoint\n")
    print("Loading Model...", end="")
    model = FidelityModel()
    model.load(path)   
    print("Done")
    # Get train and test sentences and labels
    print("Ingesting Data...", end="")
    train_data = ingest("../model_training/data/train_sentences.txt")
    train_sentences = train_data.keys()
    test_data = ingest("../model_training/data/test_sentences.txt")
    test_sentences = test_data.keys()
    print("Done")  
    # Get k nearest neighbours
    K=5
    correct = 0
    print("# MEASURING ACCURACY # ")
    progress_bar = tqdm(test_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description("Classifying With KNN")

    for test_sentence in progress_bar:
        circuits = []
        for train_sentence in train_sentences:
            circuits.append(fidelity_pqc_gen(test_sentence, train_sentence, LANGUAGE_MODEL))
        model_results = model(circuits)  # Needed for argsort in next line
        fidelities = np.array([[label, fidelity] for label, fidelity in zip(list(train_data.values()), model_results)])
        sorted_fidelities = fidelities[fidelities[:, 1].argsort()]
        highest_fidelities = sorted_fidelities[-K:]
        highest_fidelity_labels = highest_fidelities[:,0]
        label = Counter(highest_fidelity_labels).most_common(1)[0][0]
        if label==test_data[test_sentence]:
            correct += 1
    print(f"Accuracy: {round(correct*100/len(test_sentences), 2)}%")

main()