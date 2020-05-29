import pickle
shap_values = pickle.load(open('shap.pickle', 'rb'))

print(shap_values)
