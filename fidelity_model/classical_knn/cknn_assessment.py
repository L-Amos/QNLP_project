from tqdm import tqdm
from os import sys
sys.path.append("../")
from fidelity_model import FidelityModel
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
    # Ingest Data
    print("Ingesting Data...", end="")
    data = ingest(f"../model_training/data/train_sentences.txt")
    it_sentences = [key for key in list(data.keys()) if data[key]=='0']
    food_sentences = [key for key in list(data.keys()) if data[key]=='1']
    print("Done")
    # Create PQCs
    progress_bar = tqdm(it_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description("Generating IT Circuits")
    it_circuits = [sentence_pqc_gen(sentence, LANGUAGE_MODEL) for sentence in progress_bar]
    progress_bar = tqdm(food_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description("Generating Food Circuits")
    food_circuits = [sentence_pqc_gen(sentence, LANGUAGE_MODEL) for sentence in progress_bar]
    # Substitute Model Weights
    print("Subbing Model Weights...", end="")
    it_diags = model._fast_subs(it_circuits, model.weights)
    food_diags = model._fast_subs(food_circuits, model.weights)
    print("Done")
    # Run Train Sentences Through Model
    it_states = get_states(it_diags, description="Generating IT States")
    food_states = get_states(food_diags, description="Generating Food States")
    # Do the same for test sentences
    test_data = ingest(f"../model_training/data/test_sentences.txt")
    test_sentences = test_data.keys()
    progress_bar = tqdm(test_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description("Generating Test Circuits")
    test_circuits = [sentence_pqc_gen(sentence, LANGUAGE_MODEL) for sentence in progress_bar]
    test_diags = model._fast_subs(test_circuits, model.weights)
    test_states = get_states(test_diags, description="Generating Test States")
    # Get k nearest neighbours
    K=5
    correct = 0
    progress_bar = tqdm(test_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description("Classifying With KNN")
    for test_sentence, test_state in zip(progress_bar, test_states):
        fidelities = []
        for state in it_states:
            fidelities.append(['0', np.abs(test_state@state)**2])
        for state in food_states:
            fidelities.append(['1', np.abs(test_state@state)**2])
        fidelities = np.array(fidelities)
        sorted_fidelities = fidelities[fidelities[:, 1].argsort()]
        highest_fidelities = sorted_fidelities[-K:]
        highest_fidelity_labels = highest_fidelities[:,0]
        label = Counter(highest_fidelity_labels).most_common(1)[0][0]
        if label==test_data[test_sentence]:
            correct += 1
    print(f"Accuracy: {round(correct*100/len(test_sentences), 2)}%")

main()