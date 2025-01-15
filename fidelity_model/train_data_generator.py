import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction

def ingest(file_path):
    # Retrieve test sentences + parse
    with open(file_path, "r", encoding="utf8") as f:
        sentences_raw = f.readlines()
    sentences = [sentence[3:].replace('\n', '') for sentence in sentences_raw]
    labels = [sentence[0] for sentence in sentences_raw]
    data = {sentence: label for sentence, label in zip(sentences, labels)}
    # Retrieve train sentences + parse
    return data

def return_similarity(embeddings):
    SBERT_similarity = np.abs(float(SBERT_model.similarity(embeddings[0], embeddings[1])))  # |<phi|psi>|
    if SBERT_similarity > 1:  # Sometimes > 1 due to floating point errors
        SBERT_similarity = 1.0
    return SBERT_similarity

# Read Sentences
train_data = []
val_data = []
SBERT_model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
all_sentences = ingest("data/all_sentences.txt")
it_sentences = [sentence for sentence in all_sentences.keys() if all_sentences[sentence]=="0"]
cooking_sentences = [sentence for sentence in all_sentences.keys() if all_sentences[sentence]=="1"]
sentences_to_add = np.random.choice(list(all_sentences.keys()), size=80)  # For 160 training pairs

### TRAINING DATA ###
# Both same
for sentence_1 in sentences_to_add:
    ind = np.random.randint(0, len(it_sentences))
    if sentence_1 in it_sentences:
        sentence_2 = it_sentences[ind]
    else:
        sentence_2 = cooking_sentences[ind]
    embeddings = SBERT_model.encode([sentence_1, sentence_2])
    similarity = return_similarity(embeddings)
    train_data.append([sentence_1, sentence_2, similarity])
# Both Different
for sentence_1 in sentences_to_add:
    ind = np.random.randint(0, len(it_sentences))
    if sentence_1 not in it_sentences:
        sentence_2 = it_sentences[ind]
    else:
        sentence_2 = cooking_sentences[ind]
    embeddings = SBERT_model.encode([sentence_1, sentence_2])
    similarity = return_similarity(embeddings)
    train_data.append([sentence_1, sentence_2, similarity])

### VALIDATION DATA ###
# Both same
val_sentences = np.random.choice(sentences_to_add,size=20)  # For 40 validation pairs
for sentence_1 in val_sentences:
    ind = np.random.randint(0, len(it_sentences))
    if sentence_1 in it_sentences:
        sentence_2 = it_sentences[ind]
    else:
        sentence_2 = cooking_sentences[ind]
    embeddings = SBERT_model.encode([sentence_1, sentence_2])
    similarity = return_similarity(embeddings)
    val_data.append([sentence_1, sentence_2, similarity])
# Both Different
for sentence_1 in val_sentences:
    ind = np.random.randint(0, len(it_sentences))
    if sentence_1 not in it_sentences:
        sentence_2 = it_sentences[ind]
    else:
        sentence_2 = cooking_sentences[ind]
    embeddings = SBERT_model.encode([sentence_1, sentence_2])
    similarity = return_similarity(embeddings)
    val_data.append([sentence_1, sentence_2, similarity])

# Save to File
np.savetxt("data/train_data.csv",train_data, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
np.savetxt("data/val_data.csv",val_data, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
