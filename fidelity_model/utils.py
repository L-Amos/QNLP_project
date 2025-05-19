import numpy as np
import pandas as pd
from tqdm import tqdm
from lambeq import BobcatParser, RemoveCupsRewriter, StronglyEntanglingAnsatz, IQPAnsatz, Sim14Ansatz, Sim15Ansatz, Sim4Ansatz, AtomicType, bag_of_words_reader, word_sequence_reader
import tensornetwork as tn
from lambeq.backend.quantum import Ket, H, CX, Controlled, X, Id, Discard

ANSATZE = {
    0: StronglyEntanglingAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3),
    1: IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3),
    2: Sim14Ansatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3),
    3: Sim15Ansatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3),
    4: Sim4Ansatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
}


def ingest(file_path, displayname=""):
    # Retrieve test sentences + parse
    print(f"Reading {displayname}...", end="")
    csvfile = pd.read_csv(file_path)
    pairs = [[pair['sentence_1'], pair['sentence_2']] for i,pair in csvfile.iterrows()]
    labels = [pair['label'] for i,pair in csvfile.iterrows()]
    print("Done")
    return pairs, labels

def fidelity_pqc_gen(sentence_1, sentence_2, language_model=1, ansatz=0):
    # # Turn into PQCs using DisCoCat
    if language_model==1:
        parser = BobcatParser()
        remove_cups = RemoveCupsRewriter()
        sentence_1_diagram = remove_cups(parser.sentence2diagram(sentence_1))
        sentence_2_diagram = remove_cups(parser.sentence2diagram(sentence_2))
    elif language_model==2:
        sentence_1_diagram = bag_of_words_reader.sentence2diagram(sentence_1)
        sentence_2_diagram = bag_of_words_reader.sentence2diagram(sentence_2)
    elif language_model==3:
        sentence_1_diagram = word_sequence_reader.sentence2diagram(sentence_1)
        sentence_2_diagram = word_sequence_reader.sentence2diagram(sentence_2)
    ansatz = ANSATZE[ansatz]
    iqp = ansatz(sentence_1_diagram @ sentence_2_diagram)
    control = Ket(0) >> H
    fidelity_pqc =  iqp @ control
    CCX = Controlled(Controlled(X, distance=-1), distance=-1)
    fidelity_pqc >>= CX @ Id(1) >> CCX >> CX @ Id(1) >> Discard() @ Discard() @ H  # Swap Test
    return fidelity_pqc

def get_states(diags, description="Generating States"):
    results = []
    progress_bar = tqdm(diags,bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
    progress_bar.set_description(description)
    for d in progress_bar:
        result = tn.contractors.auto(*d.to_tn()).tensor
        results.append(result/np.linalg.norm(result))
    return results