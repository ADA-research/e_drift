from river.datasets import Insects


insect = Insects()
insect.download()

for x, y in insect.take(10):

    print(x,y)

