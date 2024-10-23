from sentence_transformers import SentenceTransformer, util
from state_fidelity import load_model, fidelity_test

bert_rankings = {}
my_rankings = {}
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

lambeq_model=load_model(r"C:\Users\Luke\OneDrive\Documents\Uni Stuff\Master's\NLP Project\QNLP_project\testing\model.lt")

# Retrieve test sentences + parse
with open(r"C:\Users\Luke\OneDrive\Documents\Uni Stuff\Master's\NLP Project\QNLP_project\testing\data\test_data.txt") as f:
    test_sentences_raw = f.readlines()
test_sentences = [sentence[3:].replace('\n', '') for sentence in test_sentences_raw]
# Retrieve train sentences + parse
with open(r"C:\Users\Luke\OneDrive\Documents\Uni Stuff\Master's\NLP Project\QNLP_project\testing\data\training_data.txt") as f:
    train_sentences_raw = f.readlines()
train_sentences = [sentence[3:].replace('\n', '') for sentence in train_sentences_raw]

# BERT
for train_sentence in train_sentences:
    embeddings = bert_model.encode([test_sentences[1], train_sentence])
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    bert_rankings[train_sentence] = float(similarity)

print(f"===\nBERT\n{dict(sorted(bert_rankings.items(), key=lambda x:x[1], reverse=True)[:5])}")

# LAMBEQ
for train_sentence in train_sentences:
    fidelity = fidelity_test(train_sentence, test_sentences[1], lambeq_model)[0]
    my_rankings[train_sentence] = fidelity

print(f"===\nLAMBEQ\n{dict(sorted(my_rankings.items(), key=lambda x:x[1], reverse=True)[:5])}")
