import pandas as pd


data = pd.read_csv("UMLS/umls_triples.csv")
shape = data.shape
print("Shape = {}".format(shape))
