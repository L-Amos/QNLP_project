from fidelity_model import FidelityModel
import pandas as pd
import numpy as np
from tqdm import tqdm
from lambeq import BobcatParser, RemoveCupsRewriter, StronglyEntanglingAnsatz, AtomicType, bag_of_words_reader, word_sequence_reader
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

def sentence_pqc_gen(sentence, language_model):
    # # Turn into PQCs using DisCoCat
    if language_model==1:
        parser = BobcatParser()
        remove_cups = RemoveCupsRewriter()
        sentence_diagram = remove_cups(parser.sentence2diagram(sentence))
    elif language_model==2:
        sentence_diagram =bag_of_words_reader.sentence2diagram(sentence)
    elif language_model==3:
        sentence_diagram = word_sequence_reader.sentence2diagram(sentence)
    ansatz = StronglyEntanglingAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
    circ = ansatz(sentence_diagram)
    return circ

def get_states(diags, description="Generating States"):
    results = []
    progress_bar = tqdm(diags,bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description(description)
    for d in progress_bar:
        result = tn.contractors.auto(*d.to_tn()).tensor
        results.append(result)
    return results/np.linalg.norm(results)