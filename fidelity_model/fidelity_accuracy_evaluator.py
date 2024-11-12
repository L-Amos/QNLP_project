from sentence_transformers import SentenceTransformer, util
import numpy as np
from pytket.extensions.qiskit import AerBackend
from fidelity_trainer import fidelity_pqc_gen, FidelityModel

def ingest(file_path):
    # Retrieve test sentences + parse
    with open(file_path, "r", encoding="utf8") as f:
        sentences_raw = f.readlines()
    sentences = [sentence[3:].replace('\n', '') for sentence in sentences_raw]
    labels = [sentence[0] for sentence in sentences_raw]
    data = {sentence: label for sentence, label in zip(sentences, labels)}
    # Retrieve train sentences + parse
    return data

def load_model(filename):
    """Loads a lambeq model from a checkpoint file and returns it as a TketModel object.

    Args:
        filename (str): filename of checkpoint file.

    Returns:
        FidelityModel: the loaded model.
    """
    backend = AerBackend()
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2),
        'shots': 1024
    }
    return FidelityModel.from_checkpoint(filename, backend_config=backend_config)

def get_bert_rankings(test_sentence, train_sentences, model, rank_dict={}):
    for train_sentence in train_sentences:
        embeddings = model.encode([test_sentence, train_sentence])
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        rank_dict[train_sentence] = float(similarity)
    return dict(sorted(rank_dict.items(), key=lambda x:x[1], reverse=True))  # Sort by similarity in descending order
        
def get_lambeq_rankings(test_sentence, train_sentences, model, rank_dict={}):
    diagrams = [fidelity_pqc_gen(sentence, test_sentence) for sentence in train_sentences]
    fidelities = model(diagrams)
    rank_dict = {sentence:fidelity for sentence, fidelity in zip(train_sentences, fidelities)}
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

def ndcg_eval(test_data, train_data):
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    lambeq_model=load_model("best_fidelity_model.lt")
    ndcg = []
    for test_sentence in list(test_data.keys()):
        bert_rankings = get_bert_rankings(test_sentence, list(train_data.keys()), bert_model)
        scores = score_gen(bert_rankings, test_sentence, test_data, train_data)
        idcg = np.sum([score/np.log2(i+2) for i,score in enumerate(scores.values())])
        lambeq_rankings = get_lambeq_rankings(test_sentence, list(train_data.keys()), lambeq_model)
        lambeq_scores = {sentence: scores[sentence] for sentence in lambeq_rankings}
        dcg = np.sum([score/np.log2(i+2) for i,score in enumerate(lambeq_scores.values())])
        ndcg.append(dcg/idcg)
        print(dcg/idcg)
    return np.mean(ndcg)
