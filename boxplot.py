import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

range_values = np.arange(0.001, 0.401, 0.002)
values_orange_before = np.random.choice(range_values, size = 1000, replace = True)


values_purple_before = np.append(np.random.choice(range_values, size = 800, replace = True), np.zeros(200))


range_values_after = np.arange(0.04, 0.401, 0.002)
# Step 1: Generate an array with 0 and values from 0.001 to 0.4 with steps of 0.002
values_purple_after = np.random.choice(range_values_after, size = 1000, replace = True)

# # Step 2: Create a boxplot using matplotlib
# plt.figure(figsize=(8, 6))
# plt.boxplot(values_purple_before, vert=True)
# #plt.boxplot(values_purple_after)
# plt.title('Boxplot of Values Including 0 and from 0.001 to 0.4 with Steps of 0.002')
# plt.xlabel('Value')
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.boxplot(values_purple_after, vert=True)
# #plt.boxplot(values_purple_after)
# plt.title('Boxplot of Values Including 0 and from 0.001 to 0.4 with Steps of 0.002')
# plt.xlabel('Value')
# plt.show()

plt.figure(figsize=(9, 8))
ax = sns.boxplot(data=[values_orange_before, values_purple_after], color="orange")
ax.set_xticks([0,1])
ax.set_xticklabels(["before concept drift", "after concept drift"])
plt.title("Orange class", fontsize=20)
plt.ylabel(r'$\tilde{\varepsilon}^*$ values', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("purple_class_boxplot.pdf")
plt.show()

plt.figure(figsize=(9, 8))
ax = sns.boxplot(data=[values_orange_before, values_purple_before], color="purple")
ax.set_xticks([0,1])
ax.set_xticklabels(["before concept drift", "after concept drift"])
plt.title("Purple class", fontsize=20)
plt.ylabel(r'$\tilde{\varepsilon}^*$ values', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("purple_class_boxplot.pdf")
plt.show()