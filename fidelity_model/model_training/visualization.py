import sys
sys.path.append("../")
from fidelity_model import FidelityModel
from tqdm import tqdm
from qutip import Bloch, Qobj
from utils import ingest, get_states, sentence_pqc_gen

def main():
    # Load Model
    LANGUAGE_MODEL = int(input("Which language model?\n1.\tDisCoCat\n2.\tBag of Words\n"))
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
        # Plot States
        print("Plotting States...")
        b = Bloch()
        b.add_states([Qobj(state) for state in it_states], colors=['r']*len(it_states))
        b.add_states([Qobj(state) for state in food_states], colors=['b']*len(food_states))
        b.save(path.replace("best_model.lt", f"bloch_{dataset}.png"))
        print("Done!")

main()
