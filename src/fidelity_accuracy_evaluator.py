from sentence_transformers import SentenceTransformer, util
import numpy as np
from state_fidelity import load_model, fidelity_test

ROOT_PATH = r"C:\Users\lukea\OneDrive\Documents\Uni Stuff\Master's\NLP Project\QNLP_project"

def ingest(test_path, training_path):
    # Retrieve test sentences + parse
    with open(test_path) as f:
        test_sentences_raw = f.readlines()
    test_sentences = [sentence[3:].replace('\n', '') for sentence in test_sentences_raw]
    test_labels = [sentence[0] for sentence in test_sentences_raw]
    test_data = {sentence: label for sentence, label in zip(test_sentences, test_labels)}
    # Retrieve train sentences + parse
    with open(training_path) as f:
        train_sentences_raw = f.readlines()
    train_sentences = [sentence[3:].replace('\n', '') for sentence in train_sentences_raw]
    train_labels = [sentence[0] for sentence in train_sentences_raw]
    train_data = {sentence: label for sentence, label in zip(train_sentences, train_labels)}
    return test_data, train_data

def get_bert_rankings(rank_dict, test_sentence, train_sentences, model):
    for train_sentence in train_sentences:
        embeddings = model.encode([test_sentence, train_sentence])
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        rank_dict[train_sentence] = float(similarity)
    return dict(sorted(rank_dict.items(), key=lambda x:x[1], reverse=True))  # Sort by similarity in descending order
        
def get_lambeq_rankings(rank_dict, test_sentence, train_sentences, model):
    for train_sentence in train_sentences:
        fidelity = fidelity_test(train_sentence, test_sentence, model)[0]
        rank_dict[train_sentence] = fidelity
    return dict(sorted(rank_dict.items(), key=lambda x:x[1], reverse=True))  # Sort by fidelity in descending order

def score_gen(rankings, test_sentence, test_data, train_data):
    """Generates for items in ranking - LIKELY TO NEED EDITING"""
    scores = {}
    count = 0
    for sentence in rankings:
        if test_data[test_sentence] == train_data[sentence]:
            scores[sentence] = len(rankings)-list(rankings.keys()).index(sentence)  # Set score to be the inverse of its index (so last item has score of 0)
        else:
            scores[sentence] = 0  # If sentence is part of wrong category, gets score of zero
        if count < 5:
            scores[sentence] = scores[sentence]*2  # Double top 5 scores to weight top 5
        count += 1
    return scores

def main():
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    lambeq_model=load_model(ROOT_PATH + r"\testing\model.lt")
    test_path = ROOT_PATH + r"\testing\data\test_data.txt"
    train_path = ROOT_PATH + r"\testing\data\training_data.txt"
    test_data, train_data = ingest(test_path, train_path)
    ndcg = []
    for test_sentence in list(test_data.keys()):
        bert_rankings = get_bert_rankings({}, test_sentence, list(train_data.keys()), bert_model)
        scores = score_gen(bert_rankings, test_sentence, test_data, train_data)
        idcg = np.sum([score/np.log2(i+2) for i,score in enumerate(scores.values())])
        lambeq_rankings = get_lambeq_rankings({}, test_sentence, list(train_data.keys()), lambeq_model)
        lambeq_scores = {sentence: scores[sentence] for sentence in lambeq_rankings}
        dcg = np.sum([score/np.log2(i+2) for i,score in enumerate(lambeq_scores.values())])
        ndcg.append(dcg/idcg)
    print(np.mean(ndcg))

main()