# My Fidelity Model
This folder contains code related to training my own fidelity model.

## Contents
- [Datasets and Data from Training Runs](./data/)
- [Fidelity Model Source Code](./fidelity_model.py)
- [Plotting Data from Training Runs](./plotter.ipynb)
- [Generating Good Training Datasets](./train_data_generator.py)

## About My Model
My model is built on [lambeq's quantum model](https://github.com/CQCL/lambeq/blob/main/lambeq/training/quantum_model.py). The model is trained using pairs of sentences, with the training label of each pair being the SBERT cosine similarity. 

When training, each sentence in the pair is converted to a DisCoCat diagram. The two diagrams are then composed together, and a parameterised quantum circuit (PQC) is created from this composed diagram. The model takes this PQC, adds a [swap test](https://docs.classiq.io/latest/explore/algorithms/swap_test/swap_test/) and runs the circuit using Qiskit's Aer simulator. After post-selecting the results, the model then calculates and returns the fidelity. This can be done for a single sentence pair or an array of pairs.

During training, the loss is taken to be the mean squared error between the returned fidelities and the train labels.