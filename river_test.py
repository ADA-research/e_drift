from river.datasets import synth
from river import evaluate
from river import metrics
from river import tree
from river import dummy
import pandas as pd
import itertools


gen = synth.SEA()#synth.Agrawal(classification_function=0, seed=42)

dataset = iter(gen.take(10))
print(next(dataset), "dit is dataset")

data = pd.read_csv("USP/Electricity.csv",header=None)


#retrieve labels
labels = data[8].to_list()
labels = [True if label==1 else False for label in labels]
data = data.drop(data.columns[[0, 1, 8]], axis=1) 
print(data.head())

data_list = []


def itfunc(k):
    
    for i in range(k):
        x = {idx:feature for idx,feature in enumerate(data.loc[i])}
        print(x)
        y = labels[i]

        yield x, y

data_test = iter(itfunc(10000))

model = tree.ExtremelyFastDecisionTreeClassifier(
    grace_period=100,
    delta=1e-5,
    min_samples_reevaluate=100
)


metric = metrics.MCC()

print(evaluate.progressive_val_score(data_test, model, metric))
print(model.n_drifts_detected())