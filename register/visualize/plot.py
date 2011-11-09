''' (Debug utility) Defines a set of plotting callback functions '''

import sys
import matplotlib.pyplot as plt

#===============================================================================
# Plot configuration 
#===============================================================================

IMAGE_ORIGIN=None
IMAGE_COLORMAP='gray'
IMAGE_VMIN=None
IMAGE_VMAX=None

params = {
    'axes.labelsize': 10,
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

plt.rcParams.update(params);

#===============================================================================
# Debug tools that make use of PyQt4.
#===============================================================================

def searchInspector(search):
    """
    A Qt based GUI to inspect the output of search algorithms. 
    
    Parameters
    ----------
    search: list of Register.optstepnd-array
        The output of a registration.
    """
    
    try:
        from PyQt4.QtGui import QApplication, QDialog
        from dialog import Ui_Dialog
    except Exception:
        print "Missing a required library - please install pyQt4."
        return
    
    app = QApplication(sys.argv)
    window = QDialog()
    ui = Ui_Dialog()
    ui.setupUi(window)
    ui.updateList(search)
    window.show()
    app.exec_()

#===============================================================================
# Code below needs to be cleaned up.
#===============================================================================

plt.ion()

def show():

    plt.ioff()
    plt.show()

def coordPlt(grid, buffer=10, step=5):
    """
    Plot the grid coordinates.
    """
    plt.cla()

    plt.plot(grid[1][0::step, 0::step],
             grid[0][0::step, 0::step],
             '.-b' )

    plt.plot(grid[1][0::step, 0::step].T,
             grid[0][0::step, 0::step].T,
             '.-b' )

    plt.axis( [ grid[1].max() + buffer,
                grid[1].min() - buffer,
                grid[0].max() + buffer,
                grid[0].min() - buffer],
            )
    plt.axis('off')
    plt.grid()

def featurePlt(features):
    
    for id, point in features['points'].items():
        plt.plot(point[0], point[1], 'or')


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


def featurePlot(image, template=None, warpedImage=None):
    
    plt.subplot(1,4,1)
    plt.title('I')
    plt.imshow(image.data,
               origin=IMAGE_ORIGIN,
               cmap=IMAGE_COLORMAP,
               vmin=IMAGE_VMIN,
               vmax=IMAGE_VMAX
              )
    plt.axis('off')
    featurePlt(image.features)
    
    if not template is None:
        plt.subplot(1,3,2)
        plt.title('T')
        plt.imshow(template.data,
                   origin=IMAGE_ORIGIN,
                   cmap=IMAGE_COLORMAP,
                   vmin=IMAGE_VMIN,
                   vmax=IMAGE_VMAX
                   )
        plt.axis('off')
        featurePlt(template.features)
    
    if not warpedImage is None:
        plt.subplot(1,3,3)
        plt.title('W(I;p)')
        plt.imshow(warpedImage,
                   origin=IMAGE_ORIGIN,
                   cmap=IMAGE_COLORMAP,
                   vmin=IMAGE_VMIN,
                   vmax=IMAGE_VMAX
                   )
        plt.axis('off')
        featurePlt(template.features)

def featurePlotSingle(image):
    plt.title('I')
    plt.imshow(image.data,
               cmap=IMAGE_COLORMAP,
               origin='lower',
               vmin=IMAGE_VMIN,
               vmax=IMAGE_VMAX
              )
    plt.axis('off')
    featurePlt(image.features)
    

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
