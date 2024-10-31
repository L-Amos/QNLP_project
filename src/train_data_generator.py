import numpy as np
from sentence_transformers import SentenceTransformer, util
import fidelity_accuracy_evaluator as fae

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = list(fae.ingest("testing/data/training_data.txt").keys())
data = []
for ind1, ind2 in np.random.randint(0, len(sentences), (30,2)):
    embeddings = model.encode([sentences[ind1], sentences[ind2]])
    SBERT_similarity = (float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))+1)/2
    data.append([sentences[ind1], sentences[ind2], SBERT_similarity])

np.savetxt("src/example_train.csv",data, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
