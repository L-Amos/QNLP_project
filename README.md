# Combining Statistical And Structured Quantum Methods In NLP
This is the repo for the Master's project by Luke Amos of University College London, as part of the Quantum Technologies MSc program 2024/25.

> [!IMPORTANT]
> This is the **main** branch, which houses the code for the created python module, `fidelity_model`, which is used in this project. To see the code, data and write-ups for all experiments conducted as part of this project, switch to the **experiments** branch.

## Table of Contents
- [Aim of Master's](#aim-of-the-masters)
- [Lambeq And Modifications](#lambeq-and-my-modifications-to-it)
- [Module Features](#module-features)
- [Installing](#installing)


## Aim Of The Master's[](#aim-of-the-masters)
Quantum natural language processing (QNLP) is an emergent field which uses quantum machine learning techniques for natural language tasks. It arose from the DisCoCat model fo language, which showed that sentences could be represented as operations on a collection of nouns; these computations are exactly those present in quantum computing. The DisCoCat model of language provides a highly structured approach to language, being completely based on the syntax of a sentence. 

The $k$ nearest neighbours (KNN) algorithm is a ubiquitous machine learning technique used in classical natural language processing. Being entirely statistical in nature, it is completely agnostic of the particulars of the data used, only the categories between datapoints and the distances between them. A quantum version of the algorithm (the QKNN algorithm) has been proposed and used in research, even for QNLP tasks. This provides a statistical method to solving QNLP problems.

While both DisCoCat and QKNN have been used for QNLP tasks in the past, they have never before been used in tandem. The aim of this project is to combine the structural method of DisCoCat and the statistical methods of QKNN to accomplish QNLP tasks, specifically binary and multi-class classification. The stretch goal is to use the hybrid system to create a quantum recommender system.

## Lambeq (And My Modifications To It)[](#lambeq-and-my-modifications-to-it)
Lambeq is a python package which has been created to facilitate QNLP research. Although under heavy development, it has already been used in research, and provides the backbone of the code used in this project. I have, however, had to alter it slightly.

The lambeq pipeline for training a quantum language model is as follows:
- Split a sentence into a string diagram, which is a diagrammatic representation of the sentence.
- Simplify the diagram.
- Translate the diagram into a parameterised quantum circuit (PQC).
- Train the model to pick the right parameters for each circuit.

The PQCs set a given number of qubits as 'sentence qubits', and it is the states of these qubits which represent the sentence. By default, lambeq trains by measuring these sentence qubits, and using the output as a prediction. In the most basic binary classification case, each sentence is labelled with a 0 or 1, and the aim is to train the model to output corresponding qubits in the $|0\rangle$ or $|1\rangle$ state according to the train label.

For QKNN purposes, however, it makes more sence to train a model based on *fidelity*. This is the point of the python module I have created. Instead of training the model based on the outcome of a qubit measurement, my model outputs the fidelity between two sentence qubits. Let's take the binary classification example. The training data is *pairs* of sentences, and each pair is labelled with a 0 (if the two sentences correspond to the same topic) or a 1 (if they correspond to a different topic). Then the model is trained to output sentence qubits such that those corresponding to sentences of the same topic have a fidelity of 1 (are identical states) and those corresponding to sentences of different topics have a fidelity of 0 (are orthogonal states). The hope is that, when given new sentences (using the same vocabulary), QKNN can be used to classify the new sentence state by using state fidelity as the distance measure.

A real-world use of such a model would be in a recommender system. Say you own an online bookshop, and you want to give high-quality book recommendations to your users. Using my model, you could take a blurb of a book the user has previously bought (and enjoyed) and find the $k$ most semantically similar blurbs from the database of books the user hasn't already bought, and recommend these books to the user. This relies on the assumptions that:
- The semantic similarity between blurbs reflects the similarity of the books' content.
- A user who enjoyed a book is likely to enjoy similar books.

The creation of such a recommendation system is the stretch goal of this project.

## Module Features[](#module-features)
- Custom language model based on lambeq's **Quantum NumpyModel**.
- Custom quantum trainer heavily based on lambeq's **QuantumTrainer** (modified to log the Pearson's correlation coefficient as well as losses during training).
- Utility functions for 
    - Ingesting data from CSV files.
    - Getting the sentence states themselves from the model (as the model only directly outputs fidelities between states).
    - Generating the required PQCs from sentence strings (incl. simplifying string diagrams and adding swap tests). Includes support for multiple language models and base ansatze.

## Installing[](#installing)
The module can be installed through pip as follows:
```sh
pip install git+https://github.com/L-Amos/QNLP_project
```
The following **dependencies** will be installed at the same time if not present on your system:
- NumPy
- Pandas
- Lambeq (ver 0.4.3)
- tqdm
- JAX
- TensorNetwork
