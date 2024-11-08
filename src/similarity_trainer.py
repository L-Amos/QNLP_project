"""This module trains a lambeq model to correctly assess the similarity between two sentences."""
### IMPORTS
from lambeq import BobcatParser, RemoveCupsRewriter, AtomicType, IQPAnsatz, TketModel
from lambeq.backend.converters import from_tk
from pytket import Circuit, Qubit, Bit
from pytket.extensions.qiskit import AerBackend
from pytket.circuit import OpType
from pytket.circuit.display import view_browser as draw
from sentence_transformers import SentenceTransformer, util
import numpy as np

def fidelity_circuit_gen(sentence_1, sentence_2):
    # Turn into PQCs using DisCoCat
    parser = BobcatParser()
    remove_cups = RemoveCupsRewriter()
    sentence_1_diagram = remove_cups(parser.sentence2diagram(sentence_1))
    sentence_2_diagram = remove_cups(parser.sentence2diagram(sentence_2))
    yes = sentence_1_diagram @ sentence_2_diagram
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
    # sentence_1_pqc = ansatz(sentence_1_diagram)
    # sentence_2_pqc = ansatz(sentence_2_diagram)
    yes_2 = ansatz(yes).to_tk()
    yes_2.add_qubit(Qubit("control", 0))
    draw(yes_2)
    yes_2_diag = from_tk(yes_2)
    yes_2_diag.draw()
    # # Rename bits for circuit composition
    # qubit_map = {qubit: Qubit("sentence_1", i) for i, qubit in enumerate(sentence_1_pqc.qubits)}
    # cbit_map = {bit: Bit("sentence_1_meas", i) for i, bit in enumerate(sentence_1_pqc.bits)}
    # qubit_map.update(cbit_map)
    # sentence_1_pqc.rename_units(qubit_map)
    # qubit_map = {qubit: Qubit("sentence_2", i) for i, qubit in enumerate(sentence_2_pqc.qubits)}
    # cbit_map = {bit: Bit("sentence_2_meas", i) for i, bit in enumerate(sentence_2_pqc.bits)}
    # qubit_map.update(cbit_map)
    # sentence_2_pqc.rename_units(qubit_map)
    # fidelity_pqc = sentence_1_pqc * sentence_2_pqc
    # # Add swap test
    # fidelity_cbit = Bit("fidelity_meas", 0)
    # fidelity_pqc.add_bit(fidelity_cbit)
    # control_qubit = Qubit("control", 0)
    # fidelity_pqc.add_qubit(control_qubit)
    # fidelity_pqc.add_barrier(fidelity_pqc.qubits)  # Barrier between sentence PQCs and swap test
    # measured_qubits = [op.qubits[0] for op in fidelity_pqc.commands_of_type(OpType.from_name('Measure'))]
    # cswap_qubits = [qubit for qubit in fidelity_pqc.qubits if qubit not in measured_qubits]
    # fidelity_pqc.H(control_qubit)
    # fidelity_pqc.CSWAP(*cswap_qubits)
    # fidelity_pqc.H(control_qubit)
    # fidelity_pqc.Measure(control_qubit, fidelity_cbit)
    return from_tk(fidelity_pqc)

def train():
    train_data = np.genfromtxt('src/example_train.csv', delimiter=',', dtype=None)[1:,:]
    circuits = []
    labels = train_data[:,2]
    print(labels)
    for sentence_1, sentence_2 in train_data[:,:2]:
        circuits.append(fidelity_circuit_gen(sentence_1, sentence_2))
    backend = AerBackend()
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2),
        'shots': 8192
    }
    #model = TketModel.from_diagrams(circuits, backend_config=backend_config)

train()