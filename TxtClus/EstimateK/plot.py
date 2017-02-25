import matplotlib.pyplot as plt


def stack_plot(x, x_label, y1, y2, y1_label, y2_label, ticks): 
	fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False) # create 2x1 subplots
	ax1.plot(x, y1)
	ax1.set_xticks(ticks)
	ax1.set_ylabel(y1_label)

	ax2.plot(x, y2, c="green")
	ax2.set_xticks(ticks)
	ax2.set_ylabel(y2_label)
	ax2.set_xlabel(x_label)
	plt.show()