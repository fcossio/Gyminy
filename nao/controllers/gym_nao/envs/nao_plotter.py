import matplotlib.pyplot as plt
import numpy as np
class NaoPlotter:
    def __init__ (self, title='untitled', xlabel='X', ylabel='Y'):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        self.ax.grid()
        plt.ion()

        #fig.savefig("test.png")
        plt.show()
        plt.pause(0.0000001)

    def plot(self, x=None, y=None, style='.b'):
        self.ax.plot(x,y,style)
        plt.pause(0.0000001)

    def plotPosition(self, x=None, y=None, style='.b'):
        #print('x: ',x)
        #print('y: ',y)
        self.ax.plot(x[:,0],y[:,0],'.r',x[:,1],y[:,1],'.b',x[:,2],y[:,2],'.g')
        plt.pause(0.0000001)

    def ranges(self, x1, x2, y1, y2):
        self.ax.set_xlim(x1,x2)
        self.ax.set_ylim(y1,y2)

    def resetPlot(self):
        plt.clf()
        self.fig, self.ax = plt.subplots(num=1)
        self.ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        self.ax.grid()
