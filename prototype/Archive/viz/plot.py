import matplotlib.pyplot as plt

def chart_dict(d):	
	lists = sorted(d.items()) # sorted by key, return a list of tuples
	x, y = zip(*lists) # unpack a list of pairs into two tuples
	plt.plot(x, y)
	plt.show()