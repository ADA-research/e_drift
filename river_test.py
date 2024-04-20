from river.datasets import synth
from river import evaluate
from river import metrics
from river import tree
import pandas as pd


data = pd.read_csv("USP/Electricity.csv")

print(data.head())

data = data.drop(data.columns[[0, 1]], axis=1) 

print(data.head())



model = tree.ExtremelyFastDecisionTreeClassifier(
    grace_period=100,
    delta=1e-5,
    min_samples_reevaluate=100
)


metric = metrics.Accuracy()

evaluate.progressive_val_score(data, model, metric)