### IMPORTS
from lambeq import BobcatParser, RemoveCupsRewriter, AtomicType, IQPAnsatz, TketModel
import numpy as np
from pytket.extensions.qiskit import AerBackend
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

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
    sentence1_state, sentence2_state = np.sqrt(model([sentence1_circuit, sentence2_circuit]))  # Interrogate model
    # Normalize
    # sentence1_norm = sentence1_state/np.linalg.norm(sentence1_state)
    # sentence2_norm = sentence2_state/np.linalg.norm(sentence2_state)
    return(sentence1_state, sentence2_state)

def fidelity_test(state1, state2):
    qc = QuantumCircuit(3, 1)
    qc.initialize(state1, 1)
    qc.initialize(state2, 2)
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    qc.measure(0, 0)
    sim = AerSimulator()
    job = sim.run(qc, shots=1024)
    results = job.result()
    counts = results.get_counts()
    if '0' and '1' in counts.keys():
        fidelity = counts['0']/1024 - counts['1']/1024
    else:
        fidelity = 1
    return fidelity

def main():
    model = load_model(r"C:\Users\Luke\OneDrive\Documents\Uni Stuff\Master's\NLP Project\QNLP_project\testing\model.lt")
    state1, state2 = sentence_to_state("woman prepares sauce .", "man prepares useful program .", model)
    fidelity = fidelity_test(state1, state2)
    print(fidelity)
    
main()