"""This module trains a lambeq model to correctly assess the similarity between two sentences."""
### IMPORTS
from lambeq import BobcatParser, RemoveCupsRewriter, AtomicType, IQPAnsatz, TketModel
from pytket import Circuit, Qubit, Bit
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
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
    sentence_1_pqc = ansatz(sentence_1_diagram).to_tk()
    sentence_2_pqc = ansatz(sentence_2_diagram).to_tk()
    # Rename bits for circuit composition
    qubit_map = {qubit: Qubit("sentence_1", i) for i, qubit in enumerate(sentence_1_pqc.qubits)}
    cbit_map = {bit: Bit("sentence_1_meas", i) for i, bit in enumerate(sentence_1_pqc.bits)}
    qubit_map.update(cbit_map)
    sentence_1_pqc.rename_units(qubit_map)
    qubit_map = {qubit: Qubit("sentence_2", i) for i, qubit in enumerate(sentence_2_pqc.qubits)}
    cbit_map = {bit: Bit("sentence_2_meas", i) for i, bit in enumerate(sentence_2_pqc.bits)}
    qubit_map.update(cbit_map)
    sentence_2_pqc.rename_units(qubit_map)
    fidelity_pqc = sentence_1_pqc * sentence_2_pqc
    # Add swap test
    fidelity_cbit = Bit("fidelity_meas", 0)
    fidelity_pqc.add_bit(fidelity_cbit)
    control_qubit = Qubit("control", 0)
    fidelity_pqc.add_qubit(control_qubit)
    fidelity_pqc.add_barrier(fidelity_pqc.qubits)  # Barrier between sentence PQCs and swap test
    measured_qubits = [op.qubits[0] for op in fidelity_pqc.commands_of_type(OpType.from_name('Measure'))]
    cswap_qubits = [qubit for qubit in fidelity_pqc.qubits if qubit not in measured_qubits]
    fidelity_pqc.H(control_qubit)
    fidelity_pqc.CSWAP(*cswap_qubits)
    fidelity_pqc.H(control_qubit)
    fidelity_pqc.Measure(control_qubit, fidelity_cbit)
    return fidelity_pqc

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")
# Define pair of sentences
sentence_1 = "skillful man prepares sauce ."
sentence_2 = "skillful man bakes dinner ."
# Get SBERT similarities
embeddings = model.encode([sentence_1, sentence_2])
SBERT_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
fidelity_circuit = fidelity_circuit_gen(sentence_1, sentence_2)
draw(fidelity_circuit)