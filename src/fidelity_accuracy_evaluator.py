from sentence_transformers import SentenceTransformer, util
import numpy as np
from state_fidelity import load_model, fidelity_test

ROOT_PATH = r"C:\Users\Luke\OneDrive\Documents\Uni Stuff\Master's\NLP Project\QNLP_project"

def ingest(test_path, training_path):
    # Retrieve test sentences + parse
    with open(test_path) as f:
        test_sentences_raw = f.readlines()
    test_sentences = [sentence[3:].replace('\n', '') for sentence in test_sentences_raw]
    # Retrieve train sentences + parse
    with open(training_path) as f:
        train_sentences_raw = f.readlines()
    train_sentences = [sentence[3:].replace('\n', '') for sentence in train_sentences_raw]
    return test_sentences, train_sentences

def get_bert_rankings(rank_dict, test_sentences, train_sentences, model):
    for train_sentence in train_sentences:
        embeddings = model.encode([test_sentences[1], train_sentence])
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        rank_dict[train_sentence] = float(similarity)
    return dict(sorted(rank_dict.items(), key=lambda x:x[1], reverse=True))  # Sort by similarity in descending order
        
def get_lambeq_rankings(rank_dict, test_sentences, train_sentences, model):
    for train_sentence in train_sentences:
        fidelity = fidelity_test(train_sentence, test_sentences[1], model)[0]
        rank_dict[train_sentence] = fidelity
    return dict(sorted(rank_dict.items(), key=lambda x:x[1], reverse=True))  # Sort by fidelity in descending order

def main():
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    lambeq_model=load_model(r"C:\Users\Luke\OneDrive\Documents\Uni Stuff\Master's\NLP Project\QNLP_project\testing\model.lt")
    test_path = ROOT_PATH + r"\testing\data\test_data.txt"
    train_path = ROOT_PATH + r"\testing\data\training_data.txt"
    test_sentences, train_sentences = ingest(test_path, train_path)
    bert_rankings = get_bert_rankings({}, test_sentences, train_sentences, bert_model)
    scores = {sentence:len(bert_rankings)-list(bert_rankings.keys()).index(sentence) for sentence in bert_rankings}
    idcg = np.sum([score/np.log2(i+2) for i,score in enumerate(scores.values())])
    lambeq_rankings = get_lambeq_rankings({}, test_sentences, train_sentences, lambeq_model)
    lambeq_scores = {sentence: scores[sentence] for sentence in lambeq_rankings}
    dcg = np.sum([score/np.log2(i+2) for i,score in enumerate(lambeq_scores.values())])
    print(dcg/idcg)

main()