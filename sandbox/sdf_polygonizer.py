from __future__ import division, print_function

import time
import math
from cm import *
import app
import numpy as np
import numbers
import autograff.geom as geom
import autograff.utils as utils
import autograff.plut as plut
from autograff.geom.scanlines import hatch, scanlines
from autograff.geom.simplify import dce
import autograff.geom.clipper_wrap as clip
from autograff.geom.spatial import PointQuery

from collections import namedtuple
Pass = namedtuple('Pass', 'img shader light_dir')

class Polygonizer:
    def __init__(self, render_shader_text,
                 do_shadow,
                 render_size,
                 shadow_light=[1,0,0],
                 ps_path='',
                 light_dirs = [[1, 0, 0],
                      [-1, 0, 0],
                      [0, 1, 0],
                      [0, -1, 0],
                      [0 , 0, 1],
                      [0 , 0, -1]]):

        print('loading preview')
        if ps_path:
            print('polygonizer: loading external shader')
            ps = open(ps_path).read()

        self.preview_shader = loadShader(vs, ps + render_shader_text,
                            '#define compute_color compute_color_preview\n\n')

        print('loading outline shader')
        outline_shader = loadShader(vs, ps + render_shader_text,
                            '#define compute_color compute_color_outline\n\n#define OUTLINE_PASS')

        print('loading pass shader')
        pass_shader = loadShader(vs, ps + render_shader_text,
                                    '#define compute_color compute_color_pass\n\n')

        print('loading shadow shader')
        shadow_shader = loadShader(vs, ps + render_shader_text,
                                    '#define compute_color compute_color_shadow\n\n#define SHADOW_PASS')

        print('done')
        w, h = render_size
        self.render_size = render_size

        make_img = lambda: Image(w, h, Image.BGRA)

        self.light_dirs = light_dirs

        self.passes = [Pass(make_img(), outline_shader, [1, 0, 0])]
        for ldir in self.light_dirs:
            self.passes.append( Pass(make_img(), pass_shader, ldir) )
        if do_shadow != None:
            self.has_shadow = True
            self.passes.append( Pass(make_img(), shadow_shader, shadow_light) )
        else:
            self.has_shadow = False

        # Create render targetso
        self.rt = RenderTarget(w, h, Texture.A8R8G8B8, True)
        self.rt_preview = None

        self.thresh = 0.5


    def preview(self, w, h, scene_cb, render_cb, light_dir, pass_index=-1):
        # Create preview rt if necessary
        if self.rt_preview is None:
            self.rt_preview = RenderTarget(w, h, Texture.A8R8G8B8, True)
        #return
        self.rt_preview.bind()
        clear(0, 0, 0, 1)

        shader = self.preview_shader
        if pass_index > -1:
            p = self.passes[min(pass_index, len(self.passes)-1)]
            shader = p.shader
            light_dir = p.light_dir

        data = scene_cb()
        render_cb(self.rt_preview.getWidth(),
                  self.rt_preview.getHeight(),
                  data, shader, light_dir)

        self.rt_preview.unbind()

        setOrtho(app.width(), app.height())
        #clear(1,1,1,1)
        color(1)
        self.rt_preview.draw(0, 0, self.rt_preview.getWidth(), self.rt_preview.getHeight())

    def rasterize(self, scene_cb, render_cb, verbose=False):
        if verbose:
            dprint = lambda v: print(v)
        else:
            dprint = lambda v: None

        self.rt.bind()

        dprint('Rasterizing')
        data = scene_cb()
        for i, p in enumerate(self.passes):
            clear(0,0,0,0)
            dprint('Pass %d'%(i+1))
            render_cb(self.rt.getWidth(),
                  self.rt.getHeight(),
                  data,
                  p.shader,
                  p.light_dir)

            p.img.grabFrameBuffer()
            p.img.mirror(False, True)
            # p.img.blur(2)
        dprint('Finished rasterizing')
        self.rt.unbind()

    def polygonize(self, simplify_thresh, query_radius, hatch_dist, hatch_index=1, polyfilter=lambda P: P):
        #simp = app.params['simp']
        H = []
        S = []
        #query_r = app.params['query radius']

        w, h = self.rt.getWidth(), self.rt.getHeight()
        min_sz = w / 900.
        #color(1,0,0)
        #fillCircle([query_r, query_r], query_r)

        all_ctrs = []
        for i, p in enumerate(self.passes):
            #ctrs = findContours(p.img, True)
            #ctrs = [dce.dce(ctr, 0, simp, closed=True) for ctr in ctrs]
            ctrs = dpSimplify(findContours(p.img, True, min_sz*min_sz), True, simplify_thresh)
            all_ctrs.append(ctrs)

        if hatch_index == -1 and self.has_shadow:
            hatch_index = len(self.passes) - 1

        p = None
        adapted_ctrs = []

        def add_ctrs(S, H, ctrs, k):
            if k != hatch_index:
                ctrs = [polyfilter(ctr) for ctr in ctrs if ctr.shape[1]>2]
            if k == hatch_index:
                ctrs = clip.difference(ctrs, all_ctrs[0])
                hatches = hatch(ctrs, hatch_dist, angle=0.5) #np.random.uniform(-90,90))
                #for a, b in hatches:
                #    drawLine(a, b)
                H += hatches
            if k==0:
                self.fillin = ctrs
            S += ctrs

        for k, ctrs in enumerate(all_ctrs):
            if query_radius > 0. and ctrs:
                if k==0:
                    # "seed" contour, the remaining ones are adapted
                    # Usually this would correspond with a shader that gives the whole
                    # object silhouette
                    add_ctrs(S, H, ctrs, k)
                else:
                    # Adapt remaining points
                    for i in range(len(ctrs)):
                        for j in range(ctrs[i].shape[1]):
                            pts = pt_query.query_radius(ctrs[i][:,j], query_radius)
                            if pts:
                                ctrs[i][:,j] = np.array(pts[0])
                    add_ctrs(S, H, ctrs, k)
                pt_query = PointQuery(S)
            else:
                add_ctrs(S, H, ctrs, k)

        #self.fillin = all_ctrs[0]
        return S, H

    def begin_render(self, shader, world, w, h, fov, light_dir, light_intensity=1.):
        bindShader(shader)
        enableDepthBuffer(False)
        setIdentityTransform()
        setShaderFloat2("resolution", [w, h])
        setShaderM44("invViewMatrix", np.linalg.inv(world).T) #.T) #np.linalg.inv(m))
        setShaderFloat("tanHalfFov", np.tan(geom.radians(fov/2)))
        setShaderFloat3('light', light_dir)
        setShaderFloat('light_intensity', light_intensity)
        setShaderFloat('thresh', self.thresh)

    def end_render(self):
        color(1)
        drawUVQuad(-1, -1, 2, 2)
        unbindShader()


vs = '''
void main()
{
  	gl_TexCoord[0] = gl_MultiTexCoord0;
  	gl_Position = ftransform();
}
'''

ps = '''
//------------------------------------------------------------------------------------
// A lot of the functions adapted from iq.
// http://www.iquilezles.org/
// https://www.shadertoy.com/user/iq

uniform vec2 resolution; // screen resolution
uniform float time; // current time
uniform vec2 mouse; // mouse position (screen space)
uniform float thresh;
uniform mat4 invViewMatrix;
uniform float tanHalfFov; // tan(fov/2)

uniform vec3 light;

uniform float light_intensity;

const float EPSILON = 0.01;
const float PI = 3.1415926535;
const float PI2 = PI*2.0;


float radians( float x )
{
    return PI/180.*x;
}

//------------------------------------------------------------------------------------
#pragma mark UTILS
float saturate( in float v )
{
    return clamp(v,0.0,1.0);
}

float expose( in float l, in float e )
{
    return (1.5 - exp(-l*e));
}

const vec4 lumi = vec4(0.30, 0.59, 0.11, 0);

float luminosity( in vec4 clr )
{
    return dot(clr, lumi);
}

vec4  normal_color( in vec3 n )
{
    return vec4((n*vec3(0.5)+vec3(0.5)), 1);
}

float attenuation( in float distance, in float atten )
{
    return min( 1.0/(atten*distance*distance), 1.0 );
}


// Modify these functions
float compute_scene( in vec3 p, out int mtl );
vec4 compute_color( in vec3 p, in float distance, in int mtl, in float normItCount );

//------------------------------------------------------------------------------------
#pragma mark LIGHTING

//---------------------------------------------------
// from iq. https://www.shadertoy.com/view/Xds3zN
vec3 calc_normal_i( in vec3 p )
{
    //vec3 delta = vec3( 0.004, 0.0, 0.0 );
    vec3 delta = vec3( 2./resolution.x, 0.0, 0.0 );
    int mtl;
    vec3 n;
    n.x = compute_scene( p+delta.xyz, mtl ) - compute_scene( p-delta.xyz, mtl );
    n.y = compute_scene( p+delta.yxz, mtl ) - compute_scene( p-delta.yxz, mtl );
    n.z = compute_scene( p+delta.yzx, mtl ) - compute_scene( p-delta.yzx, mtl );
    return normalize( n );
}

vec3 calc_normal( in vec3 p ) // for function f(p)
{
    float h = 1.5/resolution.x;
    int mtl;
    const vec2 k = vec2(1.,-1.);
    return normalize( k.xyy*compute_scene( p + k.xyy*h, mtl ) +
                      k.yyx*compute_scene( p + k.yyx*h, mtl ) +
                      k.yxy*compute_scene( p + k.yxy*h, mtl ) +
                      k.xxx*compute_scene( p + k.xxx*h, mtl ) );
}

vec3 calc_normal_o(vec3 pos) {
  float eps = 0.5/resolution.x;
  int mtl;
  const vec3 v1 = vec3( 1.0,-1.0,-1.0);
  const vec3 v2 = vec3(-1.0,-1.0, 1.0);
  const vec3 v3 = vec3(-1.0, 1.0,-1.0);
  const vec3 v4 = vec3( 1.0, 1.0, 1.0);

  return normalize( v1 * compute_scene( pos + v1*eps, mtl ) +
                    v2 * compute_scene( pos + v2*eps, mtl ) +
                    v3 * compute_scene( pos + v3*eps, mtl ) +
                    v4 * compute_scene( pos + v4*eps, mtl ) );
}


//---------------------------------------------------

float hard_shadow(in vec3 ro, in vec3 rd, float mint, float maxt) {
    int mtl;
    for(float t=mint; t < maxt;) {
        float h = compute_scene(ro + rd*t, mtl);
        if(h<0.001) return 0.0;
        t += h;
    }
    return 1.0;
}


//------------------------------------------------------------------------------------
#pragma mark RAY MARCHER

//This gets overridden
//#define compute_color compute_color_pass
vec3 object_color( in vec3 p, in float distance, in int mtl);

vec3 light_dir(vec3 p)
{
    return normalize(light); //mat3(invViewMatrix) * normalize(light); // - p);
}

vec4 compute_color_outline( in vec3 p, in float distance, in int mtl, in float normItCount )
{
    return vec4(1.);
}

vec4 compute_color_pass( in vec3 p, in float distance, in int mtl, in float normItCount )
{
    vec3 col = object_color(p, distance, mtl);
    //return vec4(1.,0.,0.,1.);
    //return vec4(abs(p)/10., 1.);
    vec4 it_clr = vec4(vec3(0.1+normItCount), 1.0) * 2.0;
    //return it_clr;
    //return vec4(distance*100000000);
    vec3 n = calc_normal(p);
    float d = max(0.0, dot(n, light_dir(p)))*light_intensity; //*0.5+0.5;
    //vec4 nclr = vec4(d, d, d, 1.); //(n - 1.)+0.5, 1.);
    //d = smoothstep(0, 1, d);
    float dthresh = (d>thresh)?1:0;
    return 1 - 1. * vec4(dthresh);
    //return vec4(max(0.5,luminosity(normal_color(n))) * max(0.3,ambient_occlusion1(p, n)*1.3)); // use this to debug normals
    //return vec4(max(0.5,luminosity(normal_color(n)))
    //* max(0.7,hard_shadow(p, normalize(vec3(1.,0.5,0.)), 0.01, 10.)*1.3)); // use this to debug normals
}

vec4 compute_color_shadow( in vec3 p, in float distance, in int mtl, in float normItCount )
{
    vec3 col = object_color(p, distance, mtl);
    //return vec4(1.,0.,0.,1.);
    //return vec4(abs(p)/10., 1.);
    vec4 it_clr = vec4(vec3(0.1+normItCount), 1.0) * 2.0;
    //return it_clr;
    //return vec4(distance*100000000);
    vec3 n = calc_normal(p);
    float nl = max(0., dot(n, light_dir(p)))*light_intensity;

    //return vec4(max(0.5,luminosity(normal_color(n))) * max(0.3,ambient_occlusion1(p, n)*1.3)); // use this to debug normals
    //return vec4(1. - nl*hard_shadow(p, light_dir(p), 0.1, 10.));
    return vec4(1.-col.x*hard_shadow(p, light_dir(p), 0.1, 10.));
    //return vec4(max(0.5,luminosity(normal_color(n)))
    //* max(0.7,hard_shadow(p, light_dir(p), 0.3, 10.)*1.3)); // use this to debug normals
}

vec4 compute_color_preview( in vec3 p, in float distance, in int mtl, in float normItCount )
{
    vec3 col = object_color(p, distance, mtl);
    //return vec4(1.,0.,1.,1.);
    //return vec4(abs(p)/10., 1.);
    vec4 it_clr = vec4(vec3(0.1+normItCount), 1.0) * 2.0;
    //return it_clr;
    //return vec4(distance*100000000);
    vec3 n = calc_normal(p);
    float nl = max(0.0, dot(n, light_dir(p)))*light_intensity;//*2.;
    nl = sqrt(nl);
    return vec4(col, 1.)*vec4(nl*hard_shadow(p, light_dir(p), 0.4, 10.));
    //return vec4(max(0.5,luminosity(normal_color(n))) * max(0.3,ambient_occlusion1(p, n)*1.3)); // use this to debug normals
    return vec4(max(0.5, nl)*max(0.1, luminosity(normal_color(n)))); //
    //* max(0.1,hard_shadow(p, light_dir(p), 0.1, 10.))); // use this to debug normals
}


vec4 trace_ray_enhanced(in vec3 p, in vec3 w, in vec4 bg_clr, in float pixelRadius )
{
    int mtl=0;

    float t_min = 1e-6;
    float t_max = 30.;//distance;

    float t = t_min;

    float omega = 1.1;
    float candidate_error = 1e13;
    float candidate_t = t_min;
    float previousRadius = 0.;
    float stepLength = 0.;
    float functionSign = compute_scene(p, mtl) < 0. ? -1. : 1.;

    for (int i = 0; i < 512; ++i) {
        float d = compute_scene(p + t*w, mtl);
        float signedRadius = functionSign * d;
        float radius = abs(signedRadius);

        bool sorFail = omega > 1. &&
        (radius + previousRadius) < stepLength;
        if (sorFail) {
            stepLength -= omega * stepLength;
            omega = 1.;
        } else {
            stepLength = signedRadius * omega;
        }

        previousRadius = radius;
        float error = radius / t;

        if (!sorFail && error < candidate_error) {
            candidate_t = t;
            candidate_error = error;
        }

        if (!sorFail && error < pixelRadius || t > t_max) break;

        t += stepLength;
   	}

    if ( t > t_max || candidate_error > pixelRadius )
        return bg_clr;

    return compute_color(p + t*w, t, mtl, 0); //float(i) * 1.0/float(maxIterations));//+vec3(float(i)/128.0);

    //return compute_lighting_outdoor(p + t*w, w, t, mtl, 0.);//float(i) * 1.0/float(maxIterations));//+vec3(float(i)/128.0);
}
// Ray marcher
vec4 trace_ray(in vec3 p, in vec3 w, in vec4 bg_clr, inout float distance)
{
    //    const float maxDistance = 50;//1e10;
    const int maxIterations =256;
    const float closeEnough = 1e-5;
    vec3 rp;
    int mtl;
    float t = 0;
    for (int i = 0; i < maxIterations; ++i)
    {
        rp = p+w*t;
        float d = compute_scene(rp, mtl);
        t += d;
        if (d < closeEnough)
        {
            distance = t;
            // use this to debug number of ray casts
            //return vec4(vec3(float(i)/128.0), 1.0);
//            return mtl == 0 ? vec4(vec3(float(i)/128.0), 1.0) : compute_color(rp,t,mtl);
            return compute_color(rp, t, mtl, float(i) * 1.0/float(maxIterations));//+vec3(float(i)/128.0);
        }
        else if(t > distance)
        {
            return bg_clr;//vec3(0.0);
        }


    }

    return bg_clr; //vec4(1.);//bg_clr;//vec3(0.0); // return skybox here
}

//------------------------------------------------------------------------------------
#pragma mark MAIN
void main(void)
{
    vec2 xy = gl_FragCoord.xy;

    // Primary ray origin
    vec3 p = invViewMatrix[3].xyz;
    // Primary ray direction
    vec3 w = mat3(invViewMatrix) * normalize(
                                             vec3( (xy - resolution / 2.0)*vec2(1.0,1.0), resolution.y/(-2.0*tanHalfFov))
                                             );

    float distance = 1e3;

    //vec4 clr = trace_ray(p, w, vec4(0., 0, 0, 1.), distance);
    vec4 clr = trace_ray_enhanced(p, w, vec4(0., 0, 0, 1.), (0.1/resolution.y)*sqrt(2));
    //clr = vec4(xy.x/resolution.x, xy.y/resolution.y, 0., 1.);
    //clr.xyz = pow( clr.xyz, vec3(1.0/2.2)); // gamma correction.
    //clr = vec4(abs(p)/50,1.);
    //clr.xyz = texture2D(color_image, vec2(luminosity(clr), 0.0)).xyz;

    //clr.w  = 1.0;
    gl_FragColor = clr;
}
'''
