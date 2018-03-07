import numpy as np

with open('ita-predictions','r') as f:
	pred = f.read().splitlines()

with open('ita-test-TAGS', 'r') as f:
	test = f.read().splitlines()

assert len(pred) == len(test)

correct = 0
for p,t in zip(pred, test):
	if p == t:
		correct += 1
		print(p,t)
	else:
		pass
print(correct/len(test)*100)
