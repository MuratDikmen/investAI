import pickle

shap_values = pickle.load(open('shap.pickle', 'rb'))

formatted = eval(shap_values)
print(formatted[1])
