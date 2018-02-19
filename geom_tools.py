from cm import *
import numpy as np
import autograff.geom as geom
import math

def rect_to_rect_transform(src, dst):
    src = np.array((src.l(), src.t(), src.r(), src.b())).astype(float)
    dst = np.array((dst.l(), dst.t(), dst.r(), dst.b())).astype(float)
    
    m = np.eye(3,3)
    
    sw = src[2]-src[0]
    sh = src[3]-src[1]
    dw = dst[2]-dst[0]
    dh = dst[3]-dst[1]
    
    m = geom.trans_2d([dst[0],dst[1]])
    m = np.dot(m, geom.scaling_2d([dw/sw,dh/sh]))
    m = np.dot(m, geom.trans_2d([-src[0],-src[1]]))
    
    return m

# Discretization utils
def discretize(v, step ):
    return np.round(v/step)*step

def discretize_orientation(a,b,step,start_ang=0.0):
    ''' Discretize the segment (a,b) orientation in step (degrees) increments'''
    d = b-a
    l = np.linalg.norm(d)
    ang = geom.degrees( np.arctan2(d[1], d[0]) )
    ang = discretize(ang-start_ang,step)+start_ang
    ang = geom.radians(ang%360.0)
    d = np.array([math.cos(ang),math.sin(ang)])*l
    return (a,a+d)

def discretize_line(a,b,ang=60.0,min_dist=0.01, start_ang=0.0):
    ''' Discretize orientation of a line segment specified as pair of points
        will optionally add a point if distance to target exceeds threshold min_dist'''
    min_dist *= min_dist

    a1,b1 = discretize_orientation(a,b,ang,start_ang)
    if geom.distance_squared(b,b1) < min_dist:
        return [a1,b1]

    n = int(360.0/ang)
    angs = np.linspace(0+start_ang,360+start_ang,n+1)

    pts = []
    for t in angs:
        t = geom.radians(t)
        p = b+np.array([np.cos(t),np.sin(t)])*10
        res, ins = geom.ray_intersection(a1,b1,b,p)
        #drawLine(b,p)
        #print res, ins

        if res:
            if np.any(np.isnan(ins)):
                print 'NAN!'
                print ins
            #drawCircle(ins,4)
            dist = squareDistanceToSegment(ins,a,b)
            if True: #dist > min_dist:
                pts.append((ins,dist))

    if not pts:
        return [a1,b1]

    pts.sort(key=lambda v:v[1])
    p = pts[0][0]

    return [a1,p,b]

def discretize_contour(ctr, ang, min_dist=1.0, start_ang=0.0, closed=False):
    ''' Discretize a contour (specified as a Contour or optionally as a 2 X n matrix)'''
    is_ctr = type(ctr) == Contour
    

    def conv(P):
        if is_ctr:
            return Contour(np.array(P).T, closed)
        else:
            return np.array(P).T

    if is_ctr:
        closed = ctr.closed
        P = np.array(ctr.points).T
    else:
        P = ctr.T
        
    if P.shape[0] < 2:
        print "Corrupt"
        return conv(ctr)
        
    res = []
    for a,b in zip(P,P[1:]):
        pts = discretize_line(a,b,ang,min_dist,start_ang)
        for p in pts[:-1]:
            res.append(p)
        lastp = pts[-1]
    res.append(lastp)
    
    # # remove parallel lines
    # for i in range(len(res)-2):
    #     a, b, c  = res[i], res[i+1], res[i+2]
    #     if abs(abs(np.dot(geom.normalize(b-a), geom.normalize(c-b))) - 1.0) < 1e-5:
    #         res[i] = res[i] + np.random.uniform(-2, 2, 2)
            
    if closed:
        pts = discretize_line(P[-1],P[0],ang,min_dist,start_ang)
        if len(pts) > 2 and len(res) > 2:
            isins, ins = geom.line_intersection(pts[0], pts[1], res[0], res[1])
            if isins:
                res.pop(0)
                res = [ins] + res
                res.append(pts[0])
      
    return conv(res)

def discretize_shape(S, ang, min_dist=1.0, start_ang=0.0):
    ''' Discretize a shape '''
    if type(S) != Shape and type(S) != list:
        if type(S) != Contour:
            S = Contour(S, False)
        return Shape(discretize_contour(S, ang, min_dist, start_ang))
    
    if type(S) == list:
        return [discretize_contour(P, ang, min_dist, start_ang) for P in S]
    
    Sout = Shape()
    for c in S:
        Sout.add(discretize_contour(c, ang, min_dist, start_ang))
    return Sout

def shape_fillin(s):
    ''' Returns the shape "fillin" without self intersections'''
    if type(s) == np.ndarray:
        s = Shape(Contour(s, True))
    elif type(s) == Contour:
        s = Shape(s)
    elif type(s) == Shape:
        s = s
    else:
        s = list_to_shape(s)

    box = s.boundingBox()
    box = Shape(Contour(box.corners(), True))
    return shapeIntersection(s, box)


def list_to_shape(L, close=True):
    ''' Converts a list of contours (specified as 2xN matrices) to a shape'''
    S = Shape()
    for c in L:
        S.add(Contour(c, close))
    return S

def shape_to_list(S, closed=True):
    ''' Converts a shape to a list of contours
        if specified, performs a "safe" closing operation by first closing and then cleaning up '''
    L = []
    for c in S:
        P = np.array(c.points)
        if closed:
            P = geom.close_contour(P)
            P = geom.cleanup_contour(P)
        L.append(P)
    return L

def pad_rect(rect, pad):
    return Box(rect.l() + pad,
              rect.t() + pad,
              rect.width() - pad*2,
              rect.height() - pad*2)

def rect_to_contour(rect):
    return Contour(rect.corners(), True)

def rect_in_rect(src, dst, padding=0., axis=None):
    
    dst = pad_rect(dst, padding)

    dst_w = dst.width()
    dst_h = dst.height()
    src_w = src.width()
    src_h = src.height()
    
    ratiow = dst_w/src_w
    ratioh = dst_h/src_h

    if axis==None:
        if ratiow <= ratioh:
            axis = 1
        else:
            axis = 0
    if axis==1: # fit vertically [==]
        w = dst_w
        h = src_h*ratiow
        x = dst.l()
        y = dst.t() + dst_h*0.5 - h*0.5
    else: # fit horizontally [ || ]
        w = src_w*ratioh
        h = dst_h
        
        y = dst.t()
        x = dst.l() + dst_w*0.5 - w*0.5
    
    return Box(x, y, w, h)

def rect_in_rect_transform(src, dst, padding=0., axis=None):
    fitted = rect_in_rect(src, dst, padding, axis)
    cenp_src = src.center()
    cenp_dst = fitted.center()
    
    M = np.eye(3)
    M = np.dot(M, 
               geom.trans_2d(cenp_dst - cenp_src))
    M = np.dot(M, geom.trans_2d(cenp_src))
    M = np.dot(M,
                geom.scaling_2d(np.array([
                    fitted.width()/src.width(),
                    fitted.height()/src.height()])))
    M = np.dot(M, geom.trans_2d(-cenp_src))
    return M

def rect_transform_between(rect, a, b):
    d = (b-a)
    l = np.linalg.norm(d)
    d = d/l
    cenp = rect.center()
    anchor = np.array([rect.l(), rect.t()+rect.height()/2])
    m = np.eye(3)

    m = geom.trans_2d(a)
    m = np.dot(m, geom.rot_2d(np.arctan2(d[1], d[0])))
    m = np.dot(m, geom.scaling_2d(np.ones(2)*(l/rect.width())))
    m = np.dot(m, geom.trans_2d(-anchor))
    return m

def sample_shape(shape, sample_unit=1.):
    if type(shape)==Shape:
        shape = shape_to_list(shape)
    shape = [geom.uniform_sample(ctr, sample_unit) for ctr in shape]
    return shape

def shape_center_of_mass(shape, sample_unit=1.):
    P = np.hstack( sample_shape(shape, sample_unit) )
    return np.mean(P, axis=1)

def shape_std_dev(shape, sample_unit=1.):
    P = np.hstack( sample_shape(shape, sample_unit) )
    return np.std(P, axis=1)

def shape_std_dev_side(side, shape, sample_unit=1):
    shape = sample_shape(shape, sample_unit)
    P = np.hstack(shape)
    com = np.mean(P, axis=1)
    if side:
        comp = lambda a, b: a >= b
    else:
        comp = lambda a, b: a <= b
    P = np.array([p for p in P.T if comp(p[0],com[0])]).T
    return np.std(P, axis=1)[0]
