from cm import *
import numpy as np

class Figure:
    def __init__(self):
        self.box = Box(0, 0, 300, 500)
        self.reset()
        
    def reset(self):
        self.Xs = []
        self.Ys = []
        self.markers = []

        self.colors = []
        self.axis = 'normal'
        self.cur_hue = 0.0
        self.rangex = None
        self.rangey = None

    def plot(self, *args): #X, Y, clr=None):
        if len(args) == 1:
            Y = args[0]
            X = np.linspace(0, len(Y)-1, len(Y))
            clr = None
        elif len(args) == 2:
            if len(args[0]) == len(args[1]):
                X = args[0]
                Y = args[1]
                clr = None
            else:
                Y = args[0]
                X = np.linspace(0, len(Y)-1, len(Y))
                clr = args[1]
        else:
            X, Y, clr = args

        self.Xs.append(np.array(X))
        self.Ys.append(np.array(Y))
        self.colors += [clr]
        
    def axis(self, axis):
        self.axis = axis

    def y_range(self, minv, maxv):
        self.rangey = (minv, maxv)

    def add_marker(self, p, color=[1,0,0,1.], size=3.):
        self.markers.append((p, color, size))

    def show(self, filled=False, alpha=1., histo=False):
        color(0.5)
        drawRect(self.box)
        # find min/max

        if self.rangex is None:
            minx = np.min([np.min(X) for X in self.Xs])
            maxx = np.max([np.max(X) for X in self.Xs])
        else:
            minx, maxx = self.rangex

        if self.rangey is None:
            miny = np.min([np.min(Y) for Y in self.Ys])
            maxy = np.max([np.max(Y) for Y in self.Ys])
        else:
            miny, maxy = self.rangey

        if self.axis == 'equal':
            minx = miny = min(minx, miny)
            maxx = maxy = max(maxx, maxy)
        
        rangex = maxx-minx
        rangey = maxy-miny
        if rangex == 0.0:
            rangex = 1e-20
        if rangey == 0.0:
            rangey = 1e-20


        # normalize
        def map_x(v):
            return (v - minx)/rangex
        def map_y(v):
            return (v - miny)/rangey

        Xs = [map_x(X) for X in self.Xs]
        Ys = [map_y(Y) for Y in self.Ys]

        l, t, w, h = self.box.l(), self.box.t(), self.box.width(), self.box.height()

        if miny < 0. and maxy > 0.:
            y = (0. - miny) / rangey
            y = t + h - y*h
            lineStipple(3)
            color(0.5)
            drawLine(l, y, l+w, y)
            lineStipple(0)

        hue_inc = 360. / len(Xs)
        hue = 0.

        if histo:
            for X, Y, clr in zip(Xs, Ys, self.colors):
                
                #beginVertices(LINESTRIP)
                if clr is None:
                    c = Color.hsv(hue, 1., 1.)
                    c[3] = alpha
                    color(c)
                else:
                    if len(clr) == 3:
                        clr = [clr[0], clr[1], clr[2], 1.] #np.hstack([clr, [1.]])
                    
                    color(clr)
                
                #ctr.addPoint(self.box.l(), self.box.b())
                xInc = X[1] - X[0]
                for x, y in zip(X, Y):
                    fillRect(l + x*w, t + h - y*h, w*xInc, y*h)
                    #ctr.addPoint(l + x*w, t + h - y*h)
                    #vertex(l + x*w, t + h - y*h)
                #ctr.addPoint(self.box.r(), self.box.b())

                #endVertices()
                hue = hue + hue_inc

        else:
            for X, Y, clr in zip(Xs, Ys, self.colors):
                #beginVertices(LINESTRIP)
                ctr = Contour()
                if clr is None:
                    c = Color.hsv(hue, 1., 1.)
                    c[3] = alpha
                    color(c)
                else:
                    color(clr)
                ctr.addPoint(self.box.l(), self.box.b())

                for x, y in zip(X, Y):
                    ctr.addPoint(l + x*w, t + h - y*h)
                    #vertex(l + x*w, t + h - y*h)
                ctr.addPoint(self.box.r(), self.box.b())

                if filled:
                    fill(ctr)
                else:
                    draw(ctr)
                #endVertices()
                hue = hue + hue_inc

        for p, clr, size in self.markers:
            color(clr)
            x, y = map_x(p[0]), map_y(p[1])
            fillCircle([l + x*w, t + h - y*h], size)

        return miny, maxy
        
fig = Figure()


def figure(box=Box(30, 30, 400, 300)):
    fig.reset()
    fig.box = box

def plot(*args):
    fig.plot(*args)

def add_marker(*args):
    fig.add_marker(*args)

def axis(ax):
    fig.axis(ax)

def y_range(vmin, vmax):
    fig.y_range(vmin, vmax)

def show(filled=False, alpha=1., histo=False):
    return fig.show(filled=filled, alpha=alpha, histo=histo)
