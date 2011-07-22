''' (Debug utility) Defines a set of plotting callback functions '''

import matplotlib.pyplot as plt
import numpy as np

IMAGE_ORIGIN=None
IMAGE_COLORMAP='gray'
IMAGE_VMIN=None
IMAGE_VMAX=None

params = {'axes.labelsize': 10,
          'axes.titlesize': 10,
          'figure.titlesize': 12,
          'font.size': 10,
          'font.weight':'normal',
          'text.fontsize': 10,
          'axes.fontsize': 10,
          'legend.fontsize': 11,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'figure.figsize': (12,6),
          'figure.facecolor': 'w',
          }

plt.ion()
plt.rcParams.update(params);

def show():

    plt.ioff()
    plt.show()

def coordPlt(grid, buffer=10, step=20):
    """
    Plot the grid coordinates.
    """
    plt.cla()

    plt.plot(grid[0][0::step, 0::step],
             grid[1][0::step, 0::step],
             '.-b' )

    plt.plot(grid[0][0::step, 0::step].T,
             grid[1][0::step, 0::step].T,
             '.-b' )

    plt.axis( [ grid[0].max() + buffer,
                grid[0].min() - buffer,
                grid[1].max() + buffer,
                grid[1].min() - buffer],
            )
    plt.axis('off')
    plt.grid()

def boundPlt(grid):

    xmin = grid[1].min()
    ymin = grid[0].min()

    xmax = grid[1].max()
    ymax = grid[0].max()

    plt.hlines([ymin,ymax], xmin, xmax, colors='g')
    plt.vlines([xmin, xmax], ymin, ymax, colors='g')

def warpPlot(grid, warp, _warp):

    plt.subplot(1,2,1)
    coordPlt(warp)
    boundPlt(grid)

    plt.subplot(1,2,2)
    coordPlt(_warp)
    boundPlt(grid)


def gridPlot(image, template, warpedImage, grid, warp, title):

    plt.subplot(2,3,1)
    plt.title('I')
    plt.imshow(image,
               origin=IMAGE_ORIGIN,
               cmap=IMAGE_COLORMAP,
               vmin=IMAGE_VMIN,
               vmax=IMAGE_VMAX
              )
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.title('T')
    plt.imshow(template,
               origin=IMAGE_ORIGIN,
               cmap=IMAGE_COLORMAP,
               vmin=IMAGE_VMIN,
               vmax=IMAGE_VMAX
               )
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.title('W(I;p)')
    plt.imshow(warpedImage,
               origin=IMAGE_ORIGIN,
               cmap=IMAGE_COLORMAP,
               vmin=IMAGE_VMIN,
               vmax=IMAGE_VMAX
               )
    plt.axis('off')

    plt.subplot(2,3,4)
    plt.title('I-T')
    plt.imshow(template - image,
               origin=IMAGE_ORIGIN,
               cmap=IMAGE_COLORMAP
              )
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.axis('off')
    coordPlt(warp)
    boundPlt(grid)
    plt.title('W(x;p)')

    plt.subplot(2,3,6)
    plt.title('W(I;p) - T {0}'.format(title))
    plt.imshow(template - warpedImage, 
               origin=IMAGE_ORIGIN, 
               cmap=IMAGE_COLORMAP
               )
    plt.axis('off')

    plt.draw()
