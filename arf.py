from river import forest
import numpy as np




def get_dataset(dataset_name: str):

    #retrieve features and labels of electricity dataset
    features = np.load(f"datasets/features_{dataset_name}.npy")
    labels = np.load(f"datasets/labels_{dataset_name}.npy")

    return features, labels


def classifier(features, labels):

    #initialize model
    model = forest.ARFClassifier()
    count = 0
    for feature, label in zip(features[0:2000], labels[0:2000]):
        feature = {idx:feature for idx,feature in enumerate(feature)}
        model.learn_one(feature, label)
        if model.n_drifts_detected() >0:
            print("drift", count)
        count+=1

    actual_count = model.n_drifts_detected()
    for feature, label in zip(features[2000:], labels[2000:]):
        feature = {idx:feature for idx,feature in enumerate(feature)}
        model.predict_one(feature)
        if model.n_drifts_detected() >actual_count:
            print("drift", count)
            break
        count+=1

    print("end")


def main():

    #retrieve dataset 
    dataset_name = "SEA_2_3_1"
    features, labels = get_dataset(dataset_name)
    classifier(features, labels)

if __name__ == '__main__':
    main()