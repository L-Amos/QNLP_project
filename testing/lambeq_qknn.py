from lambeq import SpacyTokeniser
from lambeq import BobcatParser
from lambeq import AtomicType, IQPAnsatz, RemoveCupsRewriter
from pytket.extensions.qiskit import tk_to_qiskit
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator

# Tokenize sentence
tokeniser = SpacyTokeniser()
sentence = "Reservoir Dogs is the best film ever made."
tokens = tokeniser.tokenise_sentence(sentence)

# Create DisCoCat diagram
parser = BobcatParser()
diagram = parser.sentence2diagram(tokens, tokenised=True)

# diagram.draw(figsize=(23,4), fontsize=12)

# Create Quantum Circuit

ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                   n_layers=2)
remove_cups = RemoveCupsRewriter()

circuit = ansatz(remove_cups(diagram))

# circuit.draw(figsize=(9, 10))

# Converting to qiskit circuit
qc = tk_to_qiskit(circuit.to_tk())
qc.draw(output='mpl')
plt.show()


