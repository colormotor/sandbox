from cm import *
import numpy as np
import autograff.geom as geom
from sandbox.geom_tools import list_to_shape
import math

def draw_normals(P, N):
    ''' Draws normals in N along points in P
        note: will draw the minimum number of columns in either P or N'''
    if type(P) == Contour:
        P = np.array(P.points)
    minn = min(P.shape[1], N.shape[1])
    P = P[:,:minn]
    N = N[:,:minn]
    Np = P + N
    Np = np.kron(P, [1,0]) + np.kron(Np, [0,1])
    drawPrimitives(Np, LINELIST)

def draw_rect(rect):
    drawRect(rect[0][0], rect[0][1], geom.rect_w(rect), geom.rect_h(rect))


def draw_shape(S, closed=True):
    if type(S) != list:
        S = [S]
    draw(list_to_shape(S, closed))

def fill_shape(S, closed=True):
    fill(list_to_shape(S, closed))
