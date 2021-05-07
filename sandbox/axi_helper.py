import axi
from cm import *
import app
import ui
import numpy as np
import random
import autograff.geom as geom
import autograff.plut as plut

V3_box = geom.make_rect(0, 0, 12, 8.5)

A3_box = geom.make_rect(0, 0, 16.93, 11.69)

A4_box = geom.make_rect(0, 0, 11.69, 8.27)

class AxiHelper:
    def __init__(self, global_vars=lambda: None):
        self.global_vars = global_vars
        try:
            global_vars.axi_device  
        except AttributeError:
            global_vars.axi_device = axi.Device()
        self.device = global_vars.axi_device

        self.move_axi_modified = []
        self.config_modified = []

    def create_device(self):
        self.global_vars.axi_device = axi.Device()
        self.device = self.global_vars.axi_device

    def add_params(self):
        app.newChild("AxiDraw")
        app.addEvent('Motors On', self.on)
        app.addEvent('Motors Off', self.off)
        app.addEvent("Pen Up", self.pen_up)
        app.addEvent("Pen Down", self.pen_down).sameLine()
        app.addEvent("home", self.go_home)
        print(A4_box)
        app.addFloat("x0", float(A4_box[0][0]), 0., 20).appendOption('v')
        app.addFloat("y0", float(A4_box[0][1]), 0., 20).appendOption('v')#.sameLine()
        app.addFloat("x1", float(A4_box[1][0]), 0., 20).appendOption('v')
        app.addFloat("y1", float(A4_box[1][1]), 0., 20).appendOption('v')#.sameLine()
        app.addEvent('draw box', self.draw_box)
        app.addFloat('DPI', 90., 1., 300.)
        app.addEvent('draw paths', self.draw)
        app.addBool('sort paths', False)

        self.move_axi_modified += [app.addFloat('pos x', 0., 0., 16)]
        self.move_axi_modified += [app.addFloat('pos y', 0., 0., 11)]

        self.config_modified += [app.addFloat('pen up pos', 60, 0., 100)]
        self.config_modified += [app.addFloat('pen down pos', 40, 0., 100)]
        
        return self

    @property
    def params(self):
        return app.params['AxiDraw']

    def on(self):
        self.create_device()
        self.device.enable_motors()

    def off(self):
        self.device.disable_motors()

    def pen_up(self):
        self.device.pen_up()

    def pen_down(self):
        self.device.pen_down()
    
    def go_home(self):
        self.device.home()
        app.setFloat('pox x', 0.)
        app.setFloat('pox y', 0.)

    def get_screen_box(self):
        dpi = self.params['DPI']
        box = (np.array([self.params['x0'], self.params['y0']]).astype(float)*dpi,
               np.array([self.params['x1'], self.params['y1']]).astype(float)*dpi)

        return box

    def clear_paths(self):
        self.paths = []

    def add_path(self, P, close=False):
        path = []
        dpi = self.params['DPI']
        P = P/dpi
        for p in P.T:
            path.append((p[0], p[1]))
        if close:
            path.append(path[0])
        self.paths.append(path)

    def add_shape(self, S, close=False):
        for P in S:
            self.add_path(P, close)

    def draw(self):
        if self.paths:
            if self.params['sort paths']:
                paths = axi.sort_paths(self.paths)
            else:
                paths = self.paths
            self.device.run_drawing(axi.Drawing(paths), True)


    def update(self):
        if np.any([p.isDirty() for p in self.move_axi_modified]):
            self.device.goto(self.params['pos x'], self.params['pos y'])
        if np.any([p.isDirty() for p in self.move_axi_modified]):
            self.device.goto(self.params['pos x'], self.params['pos y'])

        if np.any([p.isDirty() for p in self.config_modified]):
            self.device.pen_up_position = int(self.params['pen up pos'])
            self.device.pen_down_position = int(self.params['pen down pos'])
            self.device.configure()


    def draw_box(self):
        turtle = axi.Turtle()
        turtle.up()
        turtle.goto(self.params['x0'], self.params['y0'])
        turtle.down()
        turtle.goto(self.params['x1'], self.params['y0'])
        turtle.goto(self.params['x1'], self.params['y1'])
        turtle.goto(self.params['x0'], self.params['y1'])
        turtle.goto(self.params['x0'], self.params['y0'])
        turtle.up()
        self.device.run_drawing(turtle.drawing, True)

        #axi.draw(turtle.drawing)


import socket
import numpy as np

def path_to_str(P):
    ''' Convert a path to a (num_points, point sequence) tuple
        if P is a numpy array, it assumes points are columns on a 2xN matrix'''
    if type(P) == np.ndarray:
        P = P.T
    return len(P), ' '.join(['%f %f'%(p[0], p[1]) for p in P])

class AxiDrawClient:
    def __init__(self, address='localhost', port=9999):
        self.address = address
        self.port = port
        self.socket_open = False
        #self.open(address, port)

    def open(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.address, self.port)
        print('connecting to %s port %s'%server_address)
        self.sock.connect(server_address)
        self.socket_open = True

    def close(self):
        self.sock.close()
        self.socket_open = False

    def send(self, msg):
        auto_open = False
        if not self.socket_open:
            self.open()
            auto_open = True
        self.sock.sendall(msg.encode('utf-8'))
        if auto_open:
            self.close()

    def sendln(self, msg):
        self.send(msg + '\n')

    def drawing_start(self, title=''):
        self.open()
        if title:
            self.sendln('PATHCMD title ' + title)
        self.sendln('PATHCMD drawing_start')

    def drawing_end(self, raw=False):
        if raw:
            self.drawing_end_raw()
            return

        self.sendln('PATHCMD drawing_end')
        self.close()

    def drawing_end_raw(self):
        self.sendln('PATHCMD drawing_end_raw')
        self.close()

    def draw_paths(self, S, raw=False, title=''):
        self.drawing_start(title)
        for P in S:
            self.add_path(P)
        self.drawing_end(raw)

    def add_path(self, P):
        self.sendln('PATHCMD stroke %d %s'%path_to_str(P))

    def pen_up(self):
        self.sendln('PATHCMD pen_up')

    def pen_down(self):
        self.sendln('PATHCMD pen_down')

    def home(self):
        self.sendln('PATHCMD home')



class AxiClientHelper:
    def __init__(self, global_vars=lambda: None, address='localhost', port=9999):
        self.global_vars = global_vars
        self.port = port
        self.address = address
        self.create_device()
        self.title = ''

    def create_device(self):
        self.client = AxiDrawClient(self.address, self.port)  

    def add_params(self):
        app.newChild("AxiDraw")
        app.addEvent("Pen Up", self.client.pen_up)
        app.addEvent("Pen Down", self.client.pen_down).sameLine()
        app.addEvent("home", self.client.home)
        app.addEvent('draw paths', self.draw)
        app.addBool('raw drawing', True)
        app.addBool('sort paths', False)
        
        app.addFloat('DPI', 110., 1., 300.)
        
        app.addFloat("x0", float(A4_box[0][0]), 0., 20).appendOption('v')
        app.addFloat("y0", float(A4_box[0][1]), 0., 20).appendOption('v')#.sameLine()
        app.addFloat("x1", float(A4_box[1][0]), 0., 20).appendOption('v')
        app.addFloat("y1", float(A4_box[1][1]), 0., 20).appendOption('v')#.sameLine()

        return self

    @property
    def params(self):
        return app.params['AxiDraw']

    def get_screen_box(self):
        dpi = self.params['DPI']
        box = (np.array([self.params['x0'], self.params['y0']]).astype(float)*dpi,
               np.array([self.params['x1'], self.params['y1']]).astype(float)*dpi)

        return box

    def clear_paths(self):
        self.paths = []

    def add_path(self, P, close=False):
        path = []
        dpi = self.params['DPI']
        P = P/dpi
        for p in P.T:
            path.append((p[0], p[1]))
        if close:
            path.append(path[0])
        self.paths.append(path)

    def add_shape(self, S, close=False):
        for P in S:
            self.add_path(P, close)

    def draw(self):
        if self.paths:
            if self.params['sort paths']:
                paths = axi.sort_paths(self.paths)
            else:
                paths = self.paths
            self.client.draw_paths(paths, self.params['raw drawing'], title=self.title)

    def set_title(self, txt):
        self.title = txt

    def update(self):
        pass 

