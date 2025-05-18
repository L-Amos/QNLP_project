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
train_data_binary = []
val_data_binary = []
SBERT_model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
all_sentences = ingest("model_training/data/train_sentences.txt")
it_sentences = [sentence for sentence in all_sentences.keys() if all_sentences[sentence]=="0"]
cooking_sentences = [sentence for sentence in all_sentences.keys() if all_sentences[sentence]=="1"]

TEST_PAIRS = 500
VAL_PAIRS = 100

### TRAINING DATA ###
for i in range(TEST_PAIRS):
    train_sentences = np.random.choice(list(all_sentences.keys()), size=2, replace=False)
    train_embeddings = SBERT_model.encode(train_sentences, normalize_embeddings=True)
    # SBERT Similarity
    similarity = return_similarity(train_embeddings)
    pair_data = [train_sentences[0], train_sentences[1], similarity]
    train_data.append(pair_data)
    # Binary Similarity
    if np.all(np.isin(train_sentences, it_sentences)) or np.all(np.isin(train_sentences, cooking_sentences)):  # If both same category
        similarity = 1.0
    else:
        similarity = 0.0
    pair_data = [train_sentences[0], train_sentences[1], similarity]
    train_data_binary.append(pair_data)

### VALIDATION DATA ###
for i in range(VAL_PAIRS):
    val_sentences = np.random.choice(list(all_sentences.keys()), size=2, replace=False)
    val_embeddings = SBERT_model.encode(train_sentences, normalize_embeddings=True)
    # SBERT Similarity
    similarity = return_similarity(val_embeddings)
    pair_data = [val_sentences[0], val_sentences[1], similarity]
    val_data.append(pair_data)
    # Binary Similarity
    if np.all(np.isin(val_sentences, it_sentences)) or np.all(np.isin(val_sentences, cooking_sentences)):  # If both same category
        similarity = 1.0
    else:
        similarity = 0.0
    pair_data = [val_sentences[0], val_sentences[1], similarity]
    val_data_binary.append(pair_data)


# Save to File
np.savetxt("model_training/data/train_data_similarity.csv",train_data, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
np.savetxt("model_training/data/val_data_similarity.csv",val_data, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
np.savetxt("model_training/data/train_data_binary.csv",train_data_binary, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
np.savetxt("model_training/data/val_data_binary.csv",val_data_binary, delimiter=",",  fmt="%s", header="sentence_1,sentence_2,label", comments="")
