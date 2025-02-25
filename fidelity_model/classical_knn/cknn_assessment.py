from tqdm import tqdm
from os import sys
sys.path.append("../")
from fidelity_model import FidelityModel
from utils import ingest, sentence_pqc_gen, get_states
import numpy as np
from collections import Counter


def get_sentence_states(sentences, model, language_model):
    # Create PQCs
    progress_bar = tqdm(sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description("Generating Sentence Circuits")
    circuits = [sentence_pqc_gen(sentence, language_model) for sentence in progress_bar]
    # Substitute Model Weights
    print("Subbing Model Weights...", end="")
    diags = model._fast_subs(circuits, model.weights)
    print("Done")
    # Run Train Sentences Through Model
    states = get_states(diags, description="Generating States")
    return states

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
    # Get train and test sentece states
    print("# TRAIN SENTENCES #")
    train_states = get_sentence_states(train_sentences, model, LANGUAGE_MODEL)
    print("# TEST SENTENCES #")
    test_states = get_sentence_states(test_sentences, model, LANGUAGE_MODEL)    
    # Get k nearest neighbours
    K=5
    correct = 0
    print("# MEASURING ACCURACY # ")
    progress_bar = tqdm(test_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description("Classifying With KNN")
    for test_sentence, test_state in zip(progress_bar, test_states):
        fidelities = []
        for train_sentence, train_state in zip(train_sentences, train_states):
            fidelities.append([train_data[train_sentence], np.abs(test_state@train_state)**2])  # Find fidelity between test and train state
        fidelities = np.array(fidelities)  # Needed for argsort in next line
        sorted_fidelities = fidelities[fidelities[:, 1].argsort()]
        highest_fidelities = sorted_fidelities[-K:]
        highest_fidelity_labels = highest_fidelities[:,0]
        label = Counter(highest_fidelity_labels).most_common(1)[0][0]
        if label==test_data[test_sentence]:
            correct += 1
    print(f"Accuracy: {round(correct*100/len(test_sentences), 2)}%")

main()