### IMPORTS
import warnings
import os
from lambeq import BobcatParser, RemoveCupsRewriter, AtomicType, IQPAnsatz, TketModel
import numpy as np
from pytket.extensions.qiskit import AerBackend
from qiskit import QuantumCircuit

# LOAD MODEL
def load_model(filename):
    backend = AerBackend()
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2),
        'shots': 1024
    }
    return TketModel.from_checkpoint(filename, backend_config=backend_config)

def sentence_to_state(sentence1, sentence2, model):
    parser = BobcatParser()
    remove_cups = RemoveCupsRewriter()
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
    # Convert to DisCoCat Diagrams
    sentence1_diagram = remove_cups(parser.sentence2diagram(sentence1))
    sentence2_diagram = remove_cups(parser.sentence2diagram(sentence2))
    # Convert to PQCs
    sentence1_circuit = ansatz(sentence1_diagram)
    sentence2_circuit = ansatz(sentence2_diagram)
    sentence1_state, sentence2_state = model([sentence1_circuit, sentence2_circuit])  # Interrogate model
    # Normalize
    sentence1_norm = sentence1_state/np.linalg.norm(sentence1_state)
    sentence2_norm = sentence2_state/np.linalg.norm(sentence2_state)
    return(sentence1_norm, sentence2_norm)

def main():
    model = load_model(r"C:\Users\Luke\OneDrive\Documents\Uni Stuff\Master's\NLP Project\QNLP_project\src\model.lt")
    state1, state2 = sentence_to_state("woman prepares sauce .", "woman prepares tasty sauce .", model)

main()