from state_fidelity import load_model, sentences_to_circuits, fidelity_test

model=load_model(r"C:\Users\lukea\OneDrive\Documents\Uni Stuff\Master's\NLP Project\QNLP_project\testing\model.lt")

fidelity, successes = fidelity_test("man prepares program .", "woman prepares program .", model)