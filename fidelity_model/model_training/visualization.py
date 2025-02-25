import sys
sys.path.append("../")
from fidelity_model import FidelityModel
import pandas as pd
from tqdm import tqdm
from lambeq import BobcatParser, RemoveCupsRewriter, StronglyEntanglingAnsatz, AtomicType
import tensornetwork as tn
from qutip import Bloch, Qobj

def ingest(file_path):
    # Retrieve test sentences + parse
    with open(file_path, "r", encoding="utf8") as f:
        sentences_raw = f.readlines()
    sentences = [sentence[3:].replace('\n', '') for sentence in sentences_raw]
    labels = [sentence[0] for sentence in sentences_raw]
    data = {sentence: label for sentence, label in zip(sentences, labels)}
    # Retrieve train sentences + parse
    return data

def sentence_pqc_gen(sentence):
    # # Turn into PQCs using DisCoCat
    parser = BobcatParser()
    remove_cups = RemoveCupsRewriter()
    sentence_diagram = remove_cups(parser.sentence2diagram(sentence))
    ansatz = StronglyEntanglingAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
    circ = ansatz(sentence_diagram)
    return circ

def get_states(diags, model, description="Generating States"):
    results = []
    progress_bar = tqdm(diags,bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description(description)
    for d in progress_bar:
        result = tn.contractors.auto(*d.to_tn()).tensor
        result = model._normalise_vector(result)
        results.append(result)
    return results

def main():
    # Load Model
    path = input("Enter path to model checkpoint\n")
    print("Loading Model...", end="")
    model = FidelityModel()
    model.load(path)   
    print("Done")
    # Ingest Data
    for dataset in ("train", "test"):
        print(f"### {dataset.upper()} ###")
        print("Ingesting Data...", end="")
        data = ingest(f"data/{dataset}_sentences.txt")
        it_sentences = [key for key in list(data.keys()) if data[key]=='0']
        food_sentences = [key for key in list(data.keys()) if data[key]=='1']
        print("Done")
        # Create PQCs
        progress_bar = tqdm(it_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
        progress_bar.set_description("Generating IT Circuits")
        it_circuits = [sentence_pqc_gen(sentence) for sentence in progress_bar]
        progress_bar = tqdm(food_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
        progress_bar.set_description("Generating Food Circuits")
        food_circuits = [sentence_pqc_gen(sentence) for sentence in progress_bar]
        # Substitute Model Weights
        print("Subbing Model Weights...", end="")
        it_diags = model._fast_subs(it_circuits, model.weights)
        food_diags = model._fast_subs(food_circuits, model.weights)
        print("Done")
        # Run Train Sentences Through Model
        it_states = get_states(it_diags, model, description="Generating IT States")
        food_states = get_states(food_diags, model, description="Generating Food States")
        # Plot States
        print("Plotting States...")
        b = Bloch(view=[0, 0])
        b.add_states([Qobj(state) for state in it_states], colors=['r']*len(it_states))
        b.add_states([Qobj(state) for state in food_states], colors=['b']*len(food_states))
        b.save(path.replace("best_model.lt", f"bloch_{dataset}.png"))
        print("Done!")

main()
