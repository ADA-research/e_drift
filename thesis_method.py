import numpy as np
import matplotlib.pyplot as plt


def generate_dataset():
    x = np.random.rand(100,2)
    return x

def plot_dataset():
    np.random.seed(26)


    # Sample data
    #time = np.arange(1, 16)
    time_class1 = np.random.choice(np.arange(1, 41), size=20, replace=False)
    time_class2 = np.delete(np.arange(1,41), time_class1-1)
    feature2_class1 = np.random.uniform(0.532, 0.55, size=20)
    feature2_class2 = np.random.uniform(0.45, 0.53, size=len(time_class2))

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(time_class1, feature2_class1, color='orange', label='Class 1', s=200)
    plt.scatter(time_class2, feature2_class2, color='purple', label='Class 2', s=200)

    # Add lines
    plt.axhline(y=0.53, color='red', linestyle='-', linewidth=3)
    plt.axhline(y=0.505, color='green', linestyle='-', linewidth=3)

    # Labels and title
    plt.xlabel('time', fontsize=20, color='black')
    plt.ylabel('feature 2', fontsize=20, color='black')

    # Remove x-axis ticks
    plt.xticks([])

    # Set y-axis limits
    plt.ylim(0.44, 0.56)
    plt.yticks(fontsize=18)

    plt.savefig("e_drift_explanation_large.pdf")
    plt.show()
    


def plot_dataset_yellow():
    np.random.seed(9)


    # Sample data
    time_class1 = np.random.choice(np.arange(1, 41), size=20, replace=False)
    time_class2 = np.delete(np.arange(1,41), time_class1-1)
    feature2_class1 = np.random.uniform(0.50, 0.55, size=20)
    feature2_class2 = np.random.uniform(0.45, 0.50, size=len(time_class2))

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(time_class1, feature2_class1, color='orange', label='Class 1', s=200)
    plt.scatter(time_class2, feature2_class2, color='purple', label='Class 2', s=200)

    # Add lines
    plt.axhline(y=0.50, color='red', linestyle='-', linewidth=3)
    plt.axhline(y=0.505, color='green', linestyle='-', linewidth=3)

    # Add arrows from yellow points above the green line to the green line
    for i in range(len(time_class1)):
        if feature2_class1[i] <= 0.505:
            plt.annotate(
                '',
                xy=(time_class1[i], 0.505),
                xytext=(time_class1[i], feature2_class1[i]),
                arrowprops=dict(arrowstyle="->", color='blue', lw=3)
            )

    # Labels and title
    plt.xlabel('time', fontsize=20, color='black')
    plt.ylabel('feature 2', fontsize=20, color='black')

    # Remove x-axis ticks
    plt.xticks([])

    # Set y-axis limits
    plt.ylim(0.44, 0.56)
    plt.yticks(fontsize=18)

    
    plt.savefig("e_drift_explanation_misclassified_1.pdf")
    plt.show()

def plot_dataset_purple():
    np.random.seed(26)
    
    # Sample data
    time_class1 = np.random.choice(np.arange(1, 41), size=20, replace=False)
    time_class2 = np.delete(np.arange(1,41), time_class1-1)
    feature2_class1 = np.random.uniform(0.532, 0.55, size=20)
    feature2_class2 = np.random.uniform(0.45, 0.53, size=len(time_class2))

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(time_class1, feature2_class1, color='orange', label='Class 1', s=200)
    plt.scatter(time_class2, feature2_class2, color='purple', label='Class 2', s=200)

    # Add lines
    plt.axhline(y=0.53, color='red', linestyle='-', linewidth=3)
    plt.axhline(y=0.505, color='green', linestyle='-', linewidth=3)

    # Add arrows from purple points below the green line to the green line
    for i in range(len(time_class2)):
        if feature2_class2[i] > 0.505:
            plt.annotate(
                '',
                xy=(time_class2[i], 0.505),
                xytext=(time_class2[i], feature2_class2[i]),
                arrowprops=dict(arrowstyle="->", color='blue', lw=3)
            )

    # Labels and title
    plt.xlabel('time', fontsize=20, color='black')
    plt.ylabel('feature 2', fontsize=20, color='black')

    # Remove x-axis ticks
    plt.xticks([])

    # Set y-axis limits
    plt.ylim(0.44, 0.56)
    plt.yticks(fontsize=18)

    
    plt.savefig("e_drift_explanation_misclassified_2.pdf")
    plt.show()


def main():
    #set random seed
    np.random.seed(42)

    #generate random dataset
    x_values = generate_dataset()
    
    #generate plot
    #seed is 9 for small one (no concept drift)
    #seed is 26 for large one (concept drift)
    #plot_dataset()
    plot_dataset_yellow()
    plot_dataset_purple()


if __name__ == '__main__':
    main()

