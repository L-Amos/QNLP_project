"""
Tket Model
==========
Module based on a quantum backend, using `tket`.

"""
from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from lambeq import BobcatParser, RemoveCupsRewriter, IQPAnsatz, AtomicType
from lambeq.backend.quantum import Diagram
from lambeq.training.quantum_model import QuantumModel
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.circuit import OpType
from pytket import Qubit, Bit
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import StatevectorSimulator
from qiskit.visualization import plot_histogram



def fidelity_pqc_gen(sentence_1, sentence_2):
    # # Turn into PQCs using DisCoCat
    parser = BobcatParser()
    remove_cups = RemoveCupsRewriter()
    sentence_1_diagram = remove_cups(parser.sentence2diagram(sentence_1))
    sentence_2_diagram = remove_cups(parser.sentence2diagram(sentence_2))
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
    pqc = ansatz(sentence_1_diagram@sentence_2_diagram)
    return pqc
    # sentence_1_pqc = ansatz(sentence_1_diagram)
    # sentence_2_pqc = ansatz(sentence_2_diagram)
    # return sentence_1_pqc, sentence_2_pqc

class FidelityModel(QuantumModel):
    """Model based on Lambeq' `TketModel` class. Built for sentence
    fidelity testing to find semantic similarity. 
    """

    def __init__(self, device="CPU") -> None:
        """Initialise TketModel based on the `t|ket>` backend.

        """
        super().__init__()
        self.rng = np.random.default_rng()
        self.device = device

    def _randint(self, low: int = -1 << 63, high: int = (1 << 63)-1) -> int:
        return self.rng.integers(low, high, dtype=np.int64)

    def get_sentence_fidelity(self, qc):
        sentence_qubits = [qc.find_bit(qubit).index for qubit in qc.qubits]
        for gate in qc.data:
            if gate.name == 'measure':
                control_qubit = qc.find_bit(gate.qubits[0]).index
                sentence_qubits.remove(control_qubit)
        word_qubits = [qc.qubits[x] for x in range(qc.num_qubits) if x not in sentence_qubits]
        qc.remove_final_measurements()
        post_select_meas = ClassicalRegister(len(word_qubits), name='ps_meas')
        fidelity_meas = ClassicalRegister(1, name='fidelity_meas')
        control_reg = QuantumRegister(1, name='control')
        qc.add_register(post_select_meas)
        qc.add_register(fidelity_meas)
        qc.add_register(control_reg)
        qc.h(control_reg)
        qc.cswap(control_reg, *sentence_qubits)
        qc.h(control_reg)
        # Post-Selection
        qc.measure(word_qubits, post_select_meas)   
        usable = False
        sim = StatevectorSimulator(device=self.device)
        while not usable:
            job = sim.run(qc, shots=1)
            counts = job.result().get_counts()
            usable_counts = {key.split(' ')[0]: counts[key] for key in counts.keys() if '1' not in key.split(' ')[1]}
            if usable_counts:
                usable=True
        sv = job.result().get_statevector()
        return sv

    def get_diagram_output(self, diagrams: list[Diagram]) -> np.ndarray:
        """Return the prediction for each diagram using t|ket>.

        Parameters
        ----------
        diagrams : list of :py:class:`~lambeq.backend.quantum.Diagram
            The :py:class:`Circuits <lambeq.backend.quantum.Diagram>`
            to be evaluated.

        Raises
        ------
        ValueError
            If `model.weights` or `model.symbols` are not initialised.

        Returns
        -------
        np.ndarray
            Resulting array.

        """
        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`TketModel.from_diagrams()` first, '
                             'then call `initialise_weights()`, or load '
                             'from pre-trained checkpoint.')
        

        lambeq_circs = self._fast_subs(diagrams, self.weights)
        qiskit_circuits = [tk_to_qiskit(circuit.to_tk()) for circuit in lambeq_circs]
        fidelities = []
        for circ in qiskit_circuits:
            sv = self.get_sentence_fidelity(circ)
            probs = sv.probabilities_dict()
            prob_0 = sum(np.array([probs[key] for key in probs.keys() if key[0]=='0']))
            prob_1 = sum(np.array([probs[key] for key in probs.keys() if key[0]=='1']))
            fidelity = (prob_0-prob_1)/(prob_0+prob_1)
            fidelities.append(fidelity)
        return fidelities

    def forward(self, x: list[Diagram]) -> np.ndarray:
        """Perform default forward pass of a lambeq quantum model.

        In case of a different datapoint (e.g. list of tuple) or
        additional computational steps, please override this method.

        Parameters
        ----------
        x : list of :py:class:`~lambeq.backend.quantum.Diagram`
            The :py:class:`Circuits <lambeq.backend.quantum.Diagram>`
            to be evaluated.

        Returns
        -------
        np.ndarray
            Array containing model's prediction.

        """
        return self.get_diagram_output(x)