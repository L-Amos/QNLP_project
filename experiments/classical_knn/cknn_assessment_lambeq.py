from lambeq import TketModel, BobcatParser, RemoveCupsRewriter, IQPAnsatz, AtomicType
from tqdm import tqdm
from os import sys
sys.path.append("../")
from pytket.extensions.qiskit import AerBackend
from utils import ingest, sentence_pqc_gen
import numpy as np

backend = AerBackend()
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 1024
}

def accuracy(labels, results):
    correct_answers = np.sum([np.round(results[i]) == float(label) for i, label in enumerate(labels)])  # Find the number of correct answers
    accuracy = correct_answers/(2*len(labels))  # Factor of 2 because the np.round() causes double-counting
    return accuracy

LANGUAGE_MODEL = int(input("Which language model?\n1.\tDisCoCat\n2.\tBag of Words\n3.\tWord Sequence\n"))
path = input("Enter path to model checkpoint\n")
print("Loading Model...", end="")
model = TketModel.from_checkpoint(path, backend_config=backend_config)
print("Done")
# Read File
print("Ingesting Data...", end="")
test_data = ingest("../model_training/data/test_sentences.txt")
test_sentences = test_data.keys()
test_labels = list(test_data.values())
print("Done")
# Create Diagrams
progress_bar = tqdm(test_sentences, bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}")
progress_bar.set_description("Generating Sentence Circuits")
test_diags = [sentence_pqc_gen(sentence, LANGUAGE_MODEL) for sentence in progress_bar]
print("Done")
# Find Accuracy
predictions = model(test_diags)
predictions = np.round(predictions[:,0])  # Convert state vectors to binary labels
correct = 0
for i, prediction in enumerate(predictions):
    if prediction==float(test_labels[i]):
        correct += 1
print(f"Accuracy: {round(correct*100/len(test_sentences), 2)}%")
