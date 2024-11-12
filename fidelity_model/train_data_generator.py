import numpy as np
from sentence_transformers import SentenceTransformer, util
import fidelity_accuracy_evaluator as fae

# Read Sentences
data = []
SBERT_model = SentenceTransformer("all-MiniLM-L6-v2")
all_sentences = fae.ingest("data/all_sentences.txt")
it_sentences = [sentence for sentence in all_sentences.keys() if all_sentences[sentence]=="0"]
cooking_sentences = [sentence for sentence in all_sentences.keys() if all_sentences[sentence]=="1"]
sentences_to_add = np.random.choice(list(all_sentences.keys()), size=40)

### TRAINING DATA ###
# Both same
for sentence_1 in sentences_to_add:
    ind = np.random.randint(0, len(it_sentences))
    if sentence_1 in it_sentences:
        sentence_2 = it_sentences[ind]
    else:
        sentence_2 = cooking_sentences[ind]
    embeddings = SBERT_model.encode([sentence_1, sentence_2])
    SBERT_similarity = (float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))+1)/2
    data.append([sentence_1, sentence_2, SBERT_similarity])
# Both Different
for sentence_1 in sentences_to_add:
    ind = np.random.randint(0, len(it_sentences))
    if sentence_1 not in it_sentences:
        sentence_2 = it_sentences[ind]
    else:
        sentence_2 = cooking_sentences[ind]
    embeddings = SBERT_model.encode([sentence_1, sentence_2])
    SBERT_similarity = (float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))+1)/2
    if SBERT_similarity > 1:
        SBERT_similarity = 1.0
    data.append([sentence_1, sentence_2, SBERT_similarity])

### VALIDATION DATA ###
# Both same
val_sentences = np.random.choice(sentences_to_add,size=10)  # For 20 validation pairs
for sentence_1 in val_sentences:
    ind = np.random.randint(0, len(it_sentences))
    if sentence_1 in it_sentences:
        sentence_2 = it_sentences[ind]
    else:
        sentence_2 = cooking_sentences[ind]
    embeddings = SBERT_model.encode([sentence_1, sentence_2])
    SBERT_similarity = (float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))+1)/2
    data.append([sentence_1, sentence_2, SBERT_similarity])
# Both Different
for sentence_1 in val_sentences:
    ind = np.random.randint(0, len(it_sentences))
    if sentence_1 not in it_sentences:
        sentence_2 = it_sentences[ind]
    else:
        sentence_2 = cooking_sentences[ind]
    embeddings = SBERT_model.encode([sentence_1, sentence_2])
    SBERT_similarity = (float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))+1)/2
    if SBERT_similarity > 1:
        SBERT_similarity = 1.0
    data.append([sentence_1, sentence_2, SBERT_similarity])

# Save to File
np.savetxt("data/train_data.csv",data, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
np.savetxt("data/val_data.csv",data, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
