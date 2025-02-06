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

# Get SBERT Similarity
def return_similarity(embeddings):
    SBERT_similarity = np.abs(float(SBERT_model.similarity(embeddings[0], embeddings[1])))  # |<phi|psi>|
    if SBERT_similarity > 1:  # Sometimes > 1 due to floating point errors
        SBERT_similarity = 1.0
    return SBERT_similarity

# Read Sentences
train_data = []
val_data = []
SBERT_model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
all_sentences = ingest("data/train_sentences.txt")
it_sentences = [sentence for sentence in all_sentences.keys() if all_sentences[sentence]=="0"]
cooking_sentences = [sentence for sentence in all_sentences.keys() if all_sentences[sentence]=="1"]

TEST_PAIRS = 500
VAL_PAIRS = 100

### TRAINING DATA ###
for i in range(TEST_PAIRS):
    test_sentences = np.random.choice(list(all_sentences.keys()), size=2, replace=False)
    test_embeddings = SBERT_model.encode(test_sentences, normalize_embeddings=True)
    similarity = return_similarity(test_embeddings)
    pair_data = [test_sentences[0], test_sentences[1], similarity]
    train_data.append(pair_data)

### VALIDATION DATA ###
for i in range(VAL_PAIRS):
    val_sentences = np.random.choice(list(all_sentences.keys()), size=2, replace=False)
    val_embeddings = SBERT_model.encode(test_sentences, normalize_embeddings=True)
    similarity = return_similarity(val_embeddings)
    pair_data = [val_sentences[0], val_sentences[1], similarity]
    train_data.append(pair_data)

# Save to File
np.savetxt("data/train_data_alt.csv",train_data, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
np.savetxt("data/val_data_alt.csv",val_data, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
