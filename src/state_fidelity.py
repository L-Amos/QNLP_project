#%%matplotlib qt5
### IMPORTS
from lambeq import BobcatParser, RemoveCupsRewriter, AtomicType, IQPAnsatz, TketModel
from pytket.extensions.qiskit import AerBackend, tk_to_qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np


# LOAD MODEL

def load_model(filename):
    """Loads a lambeq model from a checkpoint file and returns it as a TketModel object.

    Args:
        filename (str): filename of checkpoint file.

    Returns:
        TketModel: the loaded model.
    """
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
    # Remove measurements - we are not done yet
    # sentence1_qiskit.remove_final_measurements()
    # sentence2_qiskit.remove_final_measurements()
    return sentence1_qiskit, sentence2_qiskit

def fidelity_test(sentence1, sentence2, model, draw=False):
    sentence1_circuit, sentence2_circuit = sentence_to_circuit(sentence1, sentence2, model)
    sentence1_reg = QuantumRegister(sentence1_circuit.num_qubits, "Sentence 1")
    sentence_1_meas_reg = ClassicalRegister(sentence1_circuit.num_clbits, "Sentence 1 Meas")
    sentence2_reg = QuantumRegister(sentence2_circuit.num_qubits, "Sentence 2")
    sentence_2_meas_reg = ClassicalRegister(sentence2_circuit.num_clbits, "Sentence 2 Meas")
    control_reg = QuantumRegister(1, "control")
    fidelity_meas_reg = ClassicalRegister(1, "Fidelity meas")
    qc = QuantumCircuit(control_reg, sentence1_reg, sentence_1_meas_reg, sentence2_reg, sentence_2_meas_reg, fidelity_meas_reg)
    qc = qc.compose(sentence1_circuit, sentence1_reg, sentence_1_meas_reg)
    qc = qc.compose(sentence2_circuit, sentence2_reg, sentence_2_meas_reg)
    qc.barrier()
    # Get sentence qubits
    sentence_qubits = [qc.find_bit(qubit).index for qubit in qc.qubits]
    for gate in qc.data:
        if gate.name == 'measure':
            sentence_qubits.remove(qc.find_bit(gate.qubits[0]).index)
    # Swap Test
    qc.h(control_reg)
    qc.cswap(*sentence_qubits)
    qc.h(control_reg)
    qc.measure(control_reg, fidelity_meas_reg)
    if draw:
        qc.draw('mpl')
        plt.show()
    sim = AerSimulator()
    transpiled_circ = transpile(qc, sim)
    job = sim.run(transpiled_circ, shots=2**16)
    results = job.result()
    # Post-selection
    counts = results.get_counts()
    usable_counts = {result[0]: counts[result] for result in counts if '1' not in result[1:]}
    fidelity = usable_counts.get('0', 0)/sum(usable_counts.values()) - usable_counts.get('1', 0)/sum(usable_counts.values())
    return fidelity, sum(usable_counts.values())