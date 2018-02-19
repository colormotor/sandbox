from cm import *
import numpy as np
import ui
import app
import autograff.geom as geom
import autograff.utils as utils
import os
from sandbox.geom_tools import shape_to_list, list_to_shape

def limit_vec(v, min_len, max_len):
    l = np.linalg.norm(v)
    v = v/l
    v = v*min(max(l, min_len), max_len)
    return v

def interact_tangent(index, p, theta, length, vbase, min_len=100, max_len=100 ):
    theta0 = np.arctan2(vbase[1], vbase[0])
    theta1 = theta0 + theta
    d = np.array([np.cos(theta), np.sin(theta)])*length
    b = p+d
    a = p
    color(0.5)
    drawLine(a, b)
    b2 = ui.dragger(index, b)
    d = limit_vec(b2-a, min_len, max_len)
    b = a + d
    l = np.linalg.norm(d)
    theta = np.arctan2(d[1], d[0]) #- theta0
    
    return theta, l


def interact(pts, selected, tool, dobegin=False, allow_delete=True, allow_add=True, toolbar='ab', show_draggers=True, get_modified=False, start_id=0): # TODO this get_modified is crap but need it to not break old code
    if dobegin:
        ui.begin()
    n = pts.shape[1]
    pts = np.array(pts)

    mod = False

    if show_draggers:
        for i in range(n):
            pts[:,i] = ui.dragger(i+start_id, pts[:,i])

            if ui.modified():
                mod = True
                selected = i
                allow_add = False

    delete_index = None

    if selected != None and allow_delete:        
        if app.keyPressed(app.KEY_BACKSPACE) and pts.shape[1] > 2:
            pts = np.delete(pts, selected, 1)
            delete_index = selected
            selected = None
        else:
            ui.highlightDragger(pts[:,selected])    

    if toolbar:
        tool = ui.toolbar("test", toolbar, tool)
        
    insert_index = None
    mp = app.mousePos()
    add_dist = 3

    if tool==1:
        # highlight line if in add range
        for i in range(pts.shape[1]-1):
            pa, pb = pts[:,i], pts[:,i+1]
            if geom.point_segment_distance(mp, pa, pb) < add_dist:
                ui.line(pa, pb, [1, 0, 0])

    if(tool == 1 and app.mouseClicked(0) and allow_add):
        # First check if to insert point
        ins = False
        for i in range(pts.shape[1]-1):
            pa, pb = pts[:,i], pts[:,i+1]
            if geom.point_segment_distance(mp, pa, pb) < add_dist:
                pts = np.insert(pts, i+1, mp, axis=1)
                insert_index = i+1
                ins = True
                selected = i+1
                break
        if not ins:    
            insert_index = pts.shape[1]   
            selected = pts.shape[1]
            pts = np.column_stack([pts, mp])
            
        mod = True

    if app.mouseClicked(0) and not mod:
        selected = None

    if dobegin:
        ui.end()

    if get_modified:
        return pts, selected, tool, insert_index, delete_index, mod    

    return pts, selected, tool, insert_index, delete_index

def interact_simple(pts, selected, dobegin=True):
    if dobegin:
        ui.begin()
    n = pts.shape[1]
    res = False
    for i in range(n):
        pts[:,i] = ui.dragger(i, pts[:,i])
        if ui.modified():
            selected = i
            res = True 
    if dobegin:
        ui.end()

    return pts, selected, res

class ContourMaker:
    def __init__(self):
        self.pts = np.zeros((2,0))
        self.pts_interact = np.zeros((2,0))
        self.tool = 0
        self.selected = None
        self.insert_index = None
        self.delete_index = None
        self.dobegin = True

    def clear(self):
        self.pts = np.zeros((2,0))
        self.selected = None
        self.pts_interact = np.zeros((2,0))
        
    def load(self, path=None):
        if path==None:
            path = app.openFileDialog('pkl')
        if path:
            if os.path.isfile(path):
                self.pts = utils.load_pkl(path)

    def save(self, path=None):
        if path==None:
            path = app.saveFileDialog('pkl')
        if path:
            utils.save_pkl(self.pts, path)

    def begin(self):
        ui.begin()
        self.dobegin = False
    def end(self):
        ui.end()
        self.dobegin = True

    def is_valid(self):
        return self.pts.shape[1] > 1

    
    def interact(self, insert_cb=None, delete_cb=None):
        self.pts, self.selected, self.tool, self.insert_index, self.delete_index, mod = interact(self.pts, self.selected, self.tool, self.dobegin, get_modified=True)
        if self.insert_index != None and insert_cb is not None:
            insert_cb(self.insert_index)
        if self.delete_index != None and delete_cb is not None:
            delete_cb(self.delete_index)
        return mod

    
    def interact_drag(self):
        res = False
        if app.mouseClicked(0):
            self.pts_interact = np.zeros((2,0))    
        if app.mouseDown(0):
            self.pts_interact = np.column_stack([self.pts_interact, app.mousePos()])
        if app.mouseReleased(0):
            res = True
            self.pts = geom.cleanup_contour(self.pts_interact)
            self.pts_interact = np.zeros((2,0))
        return res

    def draw(self):
        draw(self.pts, False)
        color(1,0,0,0.5)
        if self.pts_interact.shape[1]:
            draw(self.pts_interact, False)

class ShapeEdit:
    def __init__(self):
        self.shape = [] #np.zeros((2,0))]
        self.tool = 0
        self.selected = None
        self.selected_ctr = None
        self.dobegin = True

    def load_svg(self, path=None):
        if path==None:
            path = openFileDialog('svg')
        if path != None:
            svg = Shape()
            svg.loadSvg(path)
            self.shape = shape_to_list(svg)

    def load(self, path):
        if os.path.isfile(path):
            self.shape = utils.load_pkl(path)

    def save(self, path):
        utils.save_pkl(self.shape, path)

    def begin(self):
        ui.begin()
        self.dobegin = False

    def end(self):
        ui.end()
        self.dobegin = True

    def get_selected(self):
        if self.selected_ctr == None or self.selected == None:
            return None, None
        return self.selected_ctr, self.selected

    def interact(self):
        m = len(self.shape)
        self.selected_ctr = None
        for i in range(m):
            self.shape[i], self.selected, res = interact_simple(self.shape[i], self.selected)
            if res:
                self.selected_ctr = i

    def draw(self):
        for pts in self.shape:
            draw(pts, False)
        

class SemiTiedWidget:
    def __init__(self, r=100.0):
        self.params = [0.0, np.pi/2, 100.0]

    def interact(self, pos):
        r = self.params[2]
        thetas = self.params[:2]
        shift = ui.modifierShift()
        if shift:
            th0, r = ui.lengthHandle(0, [thetas[0], r], 0.0, pos, [-1000.0, 2.0], [1000.0, 500.])
            if ui.modified():
                self.params[1] = th0 + self.params[1] - self.params[0]
                self.params[0] = th0
            else:
                th1, r = ui.lengthHandle(1, [thetas[1], r], 0.0, pos, [-1000.0, 2.0], [1000.0, 500.])
                if ui.modified():
                    self.params[0] = th1 + self.params[0] - self.params[1]
                    self.params[1] = th1
        else:
            for i, theta in enumerate(thetas):
                self.params[i], r = ui.lengthHandle(i, [theta, r], 0.0, pos, [-1000.0, 2.0], [1000.0, 500.])
        self.params[2] = r

    def r(self):
        return self.params[2]

    def basis(self):
        thetas = self.params[:2]
        mat = np.array([[np.cos(theta), np.sin(theta)] for theta in thetas]).T
        return mat

    def Sigma(self, rand=False):
        r = self.params[2]
        mat = self.basis()*r
        #if rand and np.random.uniform() < 0.5:
        #    mat = np.dot(rot2d(np.pi*0.5, False), mat) #, mat)
        return np.dot(mat, mat.T)
        
    def save(self, path):
        utils.save_pkl(self.params, path)

    def load(self, path):
        try:
            self.params = utils.load_pkl(path)
        except (IOError):
            print "Could not load pkl " + path

    def make_gmm(self, X, min_rnd=1., scales=None):
        Mu = np.array(X)
        m = X.shape[1]
        if scales == None:
            scales = np.ones(m)
        Sigma = np.zeros((2,2,m))
        for i in range(m):
            r = np.random.uniform(min_rnd, 1.)*scales[i]
            Sigma[:,:,i] = self.Sigma(rand=True) * r * r
        return Mu, Sigma  

    def draw(self, pos):
        pos = np.array(pos)
        mat = self.basis()
        r = self.r()

        m33 = np.eye(3,3)
        m33[0:2,0:2] = mat
        m33[0:2,2] = pos
        pushMatrix(m33)
        color(1,0.9,0, 0.1)
        fillCircle([0,0], r)
        color(1,0,0)
        drawCircle([0,0], r)
        color(0.5)
        popMatrix()
        drawLine(pos, pos+mat[:,0]*r)
        drawLine(pos, pos+mat[:,1]*r)

    
