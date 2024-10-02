from lambeq import SpacyTokeniser

tokeniser = SpacyTokeniser()
sentence = "Bob hates Alice"
tokens = tokeniser.tokenise_sentence(sentence)
print(tokens)