"""
Tket Model
==========
Module based on a quantum backend, using `tket`.

"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from collections.abc import Callable, Iterable

import numpy
from numpy.typing import ArrayLike

from lambeq.backend import numerical_backend
from lambeq.backend.quantum import Diagram as Circuit
from lambeq.backend.tensor import Diagram
from lambeq.training.quantum_model import QuantumModel

from lambeq import BobcatParser, RemoveCupsRewriter, StronglyEntanglingAnsatz, AtomicType
from lambeq.backend.quantum import Diagram
from lambeq.training.quantum_model import QuantumModel
from lambeq.backend.converters.tk import from_tk
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.circuit import OpType
from pytket import Qubit, Bit
from qiskit import transpile
from qiskit_aer import AerSimulator

from lambeq.backend.quantum import Ket, H, CX, Controlled, X, Id, Measure, Discard

def fidelity_pqc_gen(sentence_1, sentence_2):
    # # Turn into PQCs using DisCoCat
    parser = BobcatParser()
    remove_cups = RemoveCupsRewriter()
    sentence_1_diagram = remove_cups(parser.sentence2diagram(sentence_1))
    sentence_2_diagram = remove_cups(parser.sentence2diagram(sentence_2))
    ansatz = StronglyEntanglingAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
    iqp = ansatz(sentence_1_diagram @ sentence_2_diagram)
    control = Ket(0) >> H
    fidelity_pqc =  iqp @ control
    CCX = Controlled(Controlled(X, distance=-1), distance=-1)
    fidelity_pqc >>= CX @ Id(1) >> CCX >> CX @ Id(1) >> Discard() @ Discard() @ H  # Swap Test
    return fidelity_pqc

if TYPE_CHECKING:
    from jax import numpy as jnp
class FidelityModel(QuantumModel):
    """A lambeq model for an exact classical simulation of a
    quantum pipeline."""

    def __init__(self, use_jit: bool = False) -> None:
        """Initialise an NumpyModel.

        Parameters
        ----------
        use_jit : bool, default: False
            Whether to use JAX's Just-In-Time compilation.

        """
        super().__init__()
        self.use_jit = use_jit
        self.lambdas: dict[Diagram, Callable[..., Any]] = {}

    def _get_lambda(self, diagram: Diagram) -> Callable[[Any], Any]:
        """Get lambda function that evaluates the provided diagram.

        Raises
        ------
        ValueError
            If `model.symbols` are not initialised.

        """
        from jax import jit, devices
        import tensornetwork as tn

        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`NumpyModel.from_diagrams()`.')
        if diagram in self.lambdas:
            return self.lambdas[diagram]

        def diagram_output(x: Iterable[ArrayLike]) -> ArrayLike:
            with (numerical_backend.backend('jax') as backend,
                  tn.DefaultBackend('jax')):
                sub_circuit = self._fast_subs([diagram], x)[0]
                result = tn.contractors.auto(*sub_circuit.to_tn()).tensor
                # square amplitudes to get probabilties for pure circuits
                assert isinstance(sub_circuit, Circuit)
                if not sub_circuit.is_mixed:
                    print('NOT MIXED BITCH')
                    result = backend.abs(result) ** 2
                return self._normalise_vector(result)
        self.lambdas[diagram] = jit(diagram_output, device=devices('cpu')[0])
        return self.lambdas[diagram]

    def get_diagram_output(
        self,
        diagrams: list[Diagram]
    ) -> jnp.ndarray | numpy.ndarray:
        """Return the exact prediction for each diagram.

        Parameters
        ----------
        diagrams : list of :py:class:`~lambeq.tensor.Diagram`
            The :py:class:`Circuits <lambeq.quantum.circuit.Circuit>`
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
        import tensornetwork as tn

        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`NumpyModel.from_diagrams()` first, '
                             'then call `initialise_weights()`, or load '
                             'from pre-trained checkpoint.')

        if self.use_jit:
            from jax import numpy as jnp

            lambdified_diagrams = [self._get_lambda(d) for d in diagrams]
            if hasattr(self.weights, 'filled'):
                self.weights = self.weights.filled()
            res: jnp.ndarray = jnp.array([diag_f(self.weights)
                                          for diag_f in lambdified_diagrams])
            probs = [jnp.diag(result) for result in res]
            fidelities = [prob[0] - prob[1] for prob in probs]
            return jnp.array(fidelities)

        diagrams = self._fast_subs(diagrams, self.weights)
        results = []
        for d in diagrams:
            assert isinstance(d, Circuit)
            result = tn.contractors.auto(*d.to_tn()).tensor
            # square amplitudes to get probabilties for pure circuits
            if not d.is_mixed:
                result = numpy.abs(result) ** 2
            results.append(self._normalise_vector(result))
            # Calculate Fidelity
            probs = [numpy.diag(result) for result in results]
            fidelities = [prob[0] - prob[1] for prob in probs]
        return numpy.array(fidelities)

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
