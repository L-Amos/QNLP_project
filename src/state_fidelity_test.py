#%%matplotlib qt5
### IMPORTS
from lambeq import BobcatParser, RemoveCupsRewriter, AtomicType, IQPAnsatz, TketModel
import numpy as np
from pytket.extensions.qiskit import AerBackend, tk_to_qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from lambeq.backend.quantum import Diagram as Circuit, Id, Measure
from pytket import Circuit
import matplotlib.pyplot as plt


# LOAD MODEL

def load_model(filename):
    backend = AerBackend()
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2),
        'shots': 1024
    }
    return TketModel.from_checkpoint(filename, backend_config=backend_config)

def sentence_to_circuit(sentence1, sentence2, model):
    parser = BobcatParser()
    remove_cups = RemoveCupsRewriter()
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
    # Convert to DisCoCat Diagrams
    sentence1_diagram = remove_cups(parser.sentence2diagram(sentence1))
    sentence2_diagram = remove_cups(parser.sentence2diagram(sentence2))
    # Convert to PQCs
    sentence1_circuit = ansatz(sentence1_diagram)
    sentence2_circuit = ansatz(sentence2_diagram)
    quantum_circuits = model._fast_subs([sentence1_circuit, sentence2_circuit], model.weights)
    sentence1_qiskit, sentence2_qiskit = [tk_to_qiskit(circuit.to_tk()) for circuit in quantum_circuits]
    return sentence1_qiskit, sentence2_qiskit

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
    sentence1_circuit, sentence2_circuit = sentence_to_circuit("woman prepares sauce .", "woman prepares tasty sauce .", model)
    sentence1_circuit.draw('mpl')
    sentence2_circuit.draw('mpl')
    plt.show()
    # fidelity = fidelity_test(state1, state2)
    # print(fidelity)
    
main()