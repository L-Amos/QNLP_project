"""
Tket Model
==========
Module based on a quantum backend, using `tket`.

"""
from __future__ import annotations

from typing import Any

import numpy as np

from lambeq import BobcatParser, RemoveCupsRewriter, IQPAnsatz, AtomicType
from lambeq.backend.quantum import Diagram
from lambeq.training.quantum_model import QuantumModel
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.circuit import OpType
from pytket import Qubit, Bit
from qiskit import transpile
from qiskit_aer import AerSimulator



def fidelity_pqc_gen(sentence_1, sentence_2):
    # # Turn into PQCs using DisCoCat
    parser = BobcatParser()
    remove_cups = RemoveCupsRewriter()
    sentence_1_diagram = remove_cups(parser.sentence2diagram(sentence_1))
    sentence_2_diagram = remove_cups(parser.sentence2diagram(sentence_2))
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
    fidelity_pqc = ansatz(sentence_1_diagram @ sentence_2_diagram)
    return fidelity_pqc

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
        

        
        tk_circuits = self._fast_subs(diagrams, self.weights)
        tk_circuits = [circuit.to_tk() for circuit in tk_circuits]
        fidelities = []
       
        for circuit in tk_circuits:
             # Add Swap Test
            usable_counts = {}
            fidelity_cbit = Bit("fidelity_meas", 0)
            circuit.add_bit(fidelity_cbit)
            control_qubit = Qubit("control", 0)
            circuit.add_qubit(control_qubit)
            circuit.add_barrier(circuit.qubits)  # Barrier between sentence PQCs and swap test
            measured_qubits = [op.qubits[0] for op in circuit.commands_of_type(OpType.from_name('Measure'))]
            cswap_qubits = [qubit for qubit in circuit.qubits if qubit not in measured_qubits]
            circuit.H(control_qubit)
            circuit.CSWAP(*cswap_qubits)
            circuit.H(control_qubit)
            circuit.Measure(control_qubit, fidelity_cbit)
            qc = tk_to_qiskit(circuit)
            # Measure Outcome
            sim = AerSimulator(device=self.device)
            if self.device=="GPU":
                sim.set_options(precision='single')
            transpiled_circ = transpile(qc, sim)
            while not usable_counts.values():
                job = sim.run(transpiled_circ, shots=2**17)
                results = job.result()
                # Post-selection
                counts = results.get_counts()
                try:
                    usable_counts = {result[0]: counts[result] for result in counts if '1' not in result[1:]}
                except ZeroDivisionError:
                    usable_counts = {}
            fidelity = usable_counts.get('0', 0)/sum(usable_counts.values()) - usable_counts.get('1', 0)/sum(usable_counts.values())
            fidelities.append(fidelity)
        return np.array(fidelities)

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