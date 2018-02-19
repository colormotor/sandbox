from cm import *
import app
import numpy as np

import autograff.geom as geom
import sandbox.gfx as gfx
import autograff.skeletonization as skel

class StrokeSkeleton:
    def __init__(self):
        self.strokes = []
        self.radii = []
        self.img = None
        self.G = None

    @staticmethod
    def add_params():
        config = skel.Config()
        app.newChild('Stroke Skeleton')
        app.addFloat('thresh', config.thresh, 0.0, 255)
        #app.addBool('invert', True )
        app.addFloat('blur_amt', config.blur_amt, 0, 10)
        app.addFloat('junct_eps', config.junct_eps, 1, 100)
        app.addFloat('continuity_angle', config.continuity_angle, 10., 90.)
        app.addFloat('debug stroke', 400, 0, 400)
        app.addFloat('simplify_eps', config.simplify_eps, 0, 50)

    def compute(self, img, invert=False):
        if type(img) == Image:
            img = img.mat()

        config = skel.Config()
        # Copy parameters if present
        if 'Stroke Skeleton' in app.params:
            params = app.params['Stroke Skeleton']
            for key, val in params.iteritems():
                config.__dict__[key] = val
        else:
            print "No stroke params"
        config.invert = invert
        ma = skel.medial_graph(img, config=config)
        self.G = ma[0]
        # Set 
        self.strokes, merged, self.radii = skel.get_strokes(ma)
        # Set image
        self.img = Image(self.G.graph['skel'].astype(np.uint8)*255)

    def debug_draw(self):
        if self.G == None:
            return

        P, D = skel.graph_positions_and_radii(self.G)
        # for node in self.G.nodes():
        #     color(1,0,0, 0.5)
        #     drawCircle(P[node], self.G.degree(node)*2.) #D[node])
        
        lineWidth(2.)
        for a,b in self.G.edges():
            color(0.5,0.2)
            drawLine(P[a], P[b])
        lineStipple(0)
        lineWidth(1)

        cur = 0
        nstrokes = app.params['Stroke Skeleton']['debug stroke']
        for ctr, rads in zip(self.strokes, self.radii):
            if cur > nstrokes:
                break

            color( np.hstack([np.random.uniform(0, 1, size=3), [1.]]))
            draw(ctr, False)
            color(0.5, 0.0,  0.2)
            cur = cur+1

