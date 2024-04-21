from river import dummy, tree, forest, evaluate, metrics
import pandas as pd

def itfunc(features, labels, k):
    
    for i in range(k):
        x = {idx:feature for idx,feature in enumerate(features.loc[i])}
        y = labels[i]

        yield x, y

def train_model(model, features, labels, metric):

    dataset = iter(itfunc(features, labels, len(labels)))

    output = evaluate.progressive_val_score(dataset, model, metric)
    print(output)
    print(type(output))
    print(output.get()) 

def retrieve_dataset(name: str):

    data = pd.read_csv(f"USP/{name}.csv",header=None)

    #retrieve labels
    labels = data[8].to_list()
    labels = [True if label==1 else False for label in labels]

    #retrieve features
    data = data.drop(data.columns[[0, 1, 8]], axis=1) 

    return data, labels

def main():

    #get the necessary dataset
    no_change_model = dummy.NoChangeClassifier()
    EFDT_model = tree.ExtremelyFastDecisionTreeClassifier()
    HATC_model = tree.HoeffdingAdaptiveTreeClassifier(seed=42)
    ARF_model = forest.ARFClassifier(seed=42)

    #get the necessary models
    name = "Electricity"
    features, labels = retrieve_dataset(name)
 
    #train models and evaluate 
    metric = metrics.Accuracy()

    train_model(HATC_model, features, labels, metric)




if __name__ == '__main__':
    main()