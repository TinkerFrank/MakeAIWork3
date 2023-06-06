import matplotlib.pyplot as plt
from IPython import display
plt.ion()

def realtimeplot(inputlist):

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(inputlist[0], label='training')
    plt.plot(inputlist[1], label='validation')
    plt.legend()
    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(.1)

def initialize_plot():
    plt.title('Training...')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.legend()
    plt.show()