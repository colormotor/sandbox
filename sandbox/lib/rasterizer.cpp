//
//  swrasterizer.cpp
//  swrasttest
//
//  Created by colormotor on 5/29/14.
//  Copyright (c) 2014 colormotor. All rights reserved.
//

#include "rasterizer.h"

namespace cm
{
    
#define FB_BGRA
    
#ifdef FB_BGRA
#define RED 2
#define GREEN 1
#define BLUE 0
#define ALPHA 3
#else
#define RED 0
#define GREEN 1
#define BLUE 2
#define ALPHA 3
#endif
    
    
#define MAX3( a, b, c ) ((a)>(c)?((a)>(b)?(a):(b)) : ((c)>(b)?(c):(b)))
    
#define MIN3( a, b, c ) ((a)<(c)?((a)<(b)?(a):(b)) : ((c)<(b)?(c):(b)))
    
Rasterizer::Rasterizer()
    :
    numVertices(0),
    numIndices(0)
{
    frameBuffer.depth = 0;   
}

Rasterizer::~Rasterizer()
{
    release();
}

void Rasterizer::release()
{
    SAFE_DELETE_ARRAY(frameBuffer.depth);
    numVertices = 0;
    numIndices = 0;
    frameBuffer.image.release();
}

void Rasterizer::init(int w, int h)
{
    
    frameBuffer.image = Image(w,h,Image::BGRA);
    frameBuffer.color = frameBuffer.image.mat.data; //new unsigned char[n*4];
    frameBuffer.depth = new int[w*h];
    frameBuffer.width = w;
    frameBuffer.height = h;
    frameBuffer.step = frameBuffer.image.step();
    memset(frameBuffer.color, 0, frameBuffer.step * frameBuffer.height);
    memset(frameBuffer.depth,0,w*h*sizeof(int));
    
    setProjectionMatrix(ortho(w, h));
    modelView  = arma::eye(4,4);
    
    state.fillMode = gfx::FILL_SOLID;
    state.color = V4(1,1,1,1);
    state.cullCCW = false;
    state.cullCW = false;
    state.depthOffset = 0.0001;
    state.slopeFactor = 0.0001;
    state.depthTest = true;
}

void Rasterizer::clear(float r, float g, float b, float a, bool depth)
{
    unsigned char* ptr;
    
    int n = frameBuffer.height*(frameBuffer.step/4);
    int i;
    
    int ir = r * 255;
    int ig = g * 255;
    int ib = b * 255;
    int ia = a * 255;
    
    for( ptr=frameBuffer.color, i=0; i<n; ++i, ptr+=4 )
    {
        ptr[ RED   ] = ir;
        ptr[ GREEN ] = ig;
        ptr[ BLUE  ] = ib;
        ptr[ ALPHA ] = ia;
    }
    
    if(depth)
        clearDepth();
    
    frameBuffer.image.dirty = true;
}

void Rasterizer::clearDepth(float value )
{
    unsigned int i, n;
    int* ptr;
    int v;

    n = frameBuffer.height*frameBuffer.width;
    value = value<0.0f ? 0.0f : (value>1.0f ? 1.0f : value);
    v = value * DEPTH_MAX;
    
    for( ptr=frameBuffer.depth, i=0; i<n; ++i, ++ptr )
    {
        *ptr = v;
    }
}

void Rasterizer::setModelViewMatrix( const arma::mat& m )
{
    modelView = m;
    normalMatrix = inv(m).t();
}

void Rasterizer::setProjectionMatrix( const arma::mat& m )
{
    proj = m;
    projFlipped = m;
    // negate 3'd row of projection matrix to get correct forshortening
    //projFlipped.row(3) = -projFlipped.row(3);
    
    // projFlipped.m31 = -projFlipped.m31;
    // projFlipped.m32 = -projFlipped.m32;
    // projFlipped.m33 = -projFlipped.m33;
    // projFlipped.m34 = -projFlipped.m34;
    
    // We also want our Y to go downwards
    M44 ym = arma::eye(4,4);
    ym.col(1) = -ym.col(1);
    //ym.identity();
    //ym.scale(1,-1,1);
    projFlipped = ym * projFlipped;
}

void Rasterizer::setNormals( const arma::mat& N )
{
    normalBuffer = N;
}

void Rasterizer::setColors( const arma::mat& C )
{
    colorBuffer = C;
}

void Rasterizer::setVertices( const arma::mat& V )
{
    vertexBuffer = V;
    numVertices = V.n_cols;
}

void Rasterizer::setIndices( const std::vector<unsigned int>& I )
{
    indexBuffer = I;
    numIndices = I.size();
}

void Rasterizer::setFillMode(int fillMode)
{
    state.fillMode = fillMode;
}

void Rasterizer::setCullCCW(bool flag)
{
    state.cullCCW = flag;
}

void Rasterizer::setCullCW(bool flag)
{
    state.cullCW = flag;
}

void Rasterizer::enableDepthBuffer(bool flag)
{
    state.depthTest = flag;
}

void Rasterizer::color( const arma::vec& clr )
{
    state.color = clr;
}

static V4 normalColor(const V3& n)
{
    V4 c;
    c.x = (n.x*0.5 + 0.5);
    c.y = (n.y*0.5 + 0.5);
    c.z = (n.z*0.5 + 0.5);
    c.w = 1.0;
    return c;
}

void Rasterizer::draw( const Mesh & mesh )
{
    setVertices(mesh.getVertices());
    if(mesh.numIndices())
        setIndices(mesh.getIndices());
    else
        setIndices(std::vector<unsigned int>());
    
    if(mesh.numNormals())
        setNormals(mesh.getNormals());
    else
        setNormals(arma::mat());

    if(mesh.numColors())
        setColors(mesh.getColors());
    else
        setColors(arma::mat());
    
    rasterize();
}

void Rasterizer::fill( const Shape& shape )
{
    Mesh mesh = toMesh(shape);
    draw(mesh);
}

void Rasterizer::fill( const Contour& shape )
{
    Mesh mesh = toMesh(shape);
    draw(mesh);
}


void Rasterizer::rasterize()
{
    std::vector<Rasterizer::Vertex> verts(numVertices);
        
    // transform
    for( int i = 0; i < numVertices; i++ )
    {
        // vertex shader could go here
        Rasterizer::Vertex & v = verts[i];
        v.v = V3(vertexBuffer.col(i));
        v.viewp = mul(modelView, v.v);
        
        v.projp = projFlipped * V4(v.viewp, 1);
        V4 clr;
        if(colorBuffer.n_cols)
            clr = colorBuffer.col(i);
        else
            clr = state.color;
        
        if(normalBuffer.n_cols)
        {
            v.n = mul(normalMatrix, normalBuffer.col(i));
            v.clr = normalColor(v.n);
        }
        else
        {
            v.n = V3(0,0,0);
            v.clr = clr;
        }
    }
    
    rasterize(verts);
    frameBuffer.image.dirty = true;
}
    
void Rasterizer::rasterize( const std::vector<Rasterizer::Vertex> verts )
{
    if(indexBuffer.size())
    {
        for( int i = 0; i < numIndices; i+=3 )
        {
            rasterizeTriangle(verts[indexBuffer[i]],
                              verts[indexBuffer[i+1]],
                              verts[indexBuffer[i+2]]);
        }
    }
    else
    {
        for( int i = 0; i < numVertices; i+=3 )
        {
            rasterizeTriangle(verts[i],
                              verts[i+1],
                              verts[i+2]);
        }
        /*
        for( int i = 0; i < numVertices; i+=3 )
        {
            const Rasterizer::Vertex & va = verts[i];
            const Rasterizer::Vertex & vb = verts[i+1];
            const Rasterizer::Vertex & vc = verts[i+2];
            
            rasterizeSegment(va,vb);
            rasterizeSegment(vb,vc);
            rasterizeSegment(vc,va);
        }*/
    }
}
    
bool Rasterizer::testFrag( const Rasterizer::Fragment & f, int x, int y )
{
    int w = frameBuffer.width;
    int h = frameBuffer.height;
    int * buf = frameBuffer.depth;
    
    static const int ofs[8][2] = {{-1,0},{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1}};
    int d = buf[y*w+x];
    
    for( int i = 0; i < 8; i++ )
    {
        int xx = x+ofs[i][0];
        int yy = y+ofs[i][1];
        if(xx>=0 && xx < w && yy >= 0 && yy < h )
            d = std::max(d,buf[yy*w+xx]);
    }
    
    
    return f.d <= d;
}

Rasterizer::Vertex Rasterizer::makeVertex( const V3& pos, const V3& normal, const V4& clr)
{
    Rasterizer::Vertex v;
    v.v = pos;
    v.viewp = mul(modelView, v.v);
    v.projp = projFlipped * V4(v.viewp, 1.);
    v.clr = clr;
    v.n = mul(normalMatrix, normal);
    return v;
}

arma::mat Rasterizer::getRasterizedSegmentPoints( const arma::vec& a_, const arma::vec& b_, bool ztest )
{
    V3 a;
    V3 b;
    if(a.n_cols < 3)
    {
        a = V3(V2(a), 0.);
        b = V3(V2(b), 0.);
    }
    else
    {
        a = a;
        b = b;
    }

    V3 normal = V3(0, 0, 1);
    int n = getRasterizedSegmentPoints( rastX, rastY, rastVisible, makeVertex(a, normal), makeVertex(b, normal), ztest );
    arma::mat res;

    if(n)
    {
        res = arma::zeros(3, n);
        for( int i = 0; i < n; i++ )
        {
            res.col(i) = arma::vec({(double)rastX[i], (double)rastY[i], (double)rastVisible[i]});
        }
    }
    
    return res;
}

int Rasterizer::getRasterizedSegmentPoints( int * X, int * Y, int * visible, const Rasterizer::Vertex & va,
                                const Rasterizer::Vertex & vb,
                                bool ztest )
{
    int w = frameBuffer.width;
    int h = frameBuffer.height;
    
#define sign(v) (v>0)?1:-1
    
    V4 p1 = va.projp;
    V4 p2 = vb.projp;
    p1.w = 1.0/p1.w;
    p2.w = 1.0/p2.w;
    
    p1.x = ((p1.x*p1.w)*0.5+0.5)*frameBuffer.width;
    p2.x = ((p2.x*p2.w)*0.5+0.5)*frameBuffer.width;
    
    p1.y = ((p1.y*p1.w)*0.5+0.5)*frameBuffer.height;
    p2.y = ((p2.y*p2.w)*0.5+0.5)*frameBuffer.height;
    
    // perspective texture mapping. (?)
    // for( i=0; i<MAX_TEXTURES; ++i )
    //         {
    //             v.s[i] = (A.s[i]*a + B.s[i]*b + C.s[i]*c) * v.w;
    //             v.t[i] = (A.t[i]*a + B.t[i]*b + C.t[i]*c) * v.w;
    //         }

    // depth slope calc as defined in http://www.glprogramming.com/red/chapter06.html
    
    V3 d = vb.viewp-va.viewp; //.xyz()
    float m = 0.0f;
    if(d.x!=0)
        m = fabs(d.z/d.x);
    if(d.y!=0)
        m = std::max(m,(float)fabs(d.z/d.y));
    m*=state.slopeFactor*0.001;
    float eps = state.depthOffset;//+m;
    
    int z1 = DEPTH_MAX_HALF - ((p1.z+eps)*DEPTH_MAX_HALF*p1.w);
    int z2 = DEPTH_MAX_HALF - ((p2.z+eps)*DEPTH_MAX_HALF*p2.w);
    
    int x1 = p1.x;
    int x2 = p2.x;
    int y1 = p1.y;
    int y2 = p2.y;
    
    int i = 0;
    
    int dx, dy, inx, iny, e;
    
    dx = x2 - x1;
    dy = y2 - y1;
    
    float ddx = dx;
    float ddy = dy;
    
    // not a segment
    if(dx == 0 && dy == 0)
        return 0;
    
    inx = sign(dx);
    iny = sign(dy);
    
    dx = abs(dx);
    dy = abs(dy);
    
    float t,z;
    
    int x = x1;
    int y = y1;
    
    bool iter = true;
    
    Fragment f;
    f.clr = V4(0,0,0,1);
    
    bool clip = false;
    
    if(dx >= dy)
    {
        dy <<= 1;
        e = dy - dx;
        dx <<= 1;
        while (iter)
        {
            if(x==x2)
                iter = false;
            
            t = ((float)(x-x1)*ddx)/(ddx*ddx);
            f.d = (float)z1 + ((float)z2 - z1)*t;
            
            X[i] = x;
            Y[i] = y;
            visible[i] = 0;
            
            if(x>=0 && x < w && y >= 0 && y < h )
            {
                if(!ztest || testFrag(f,x,y))
                {
                    int ind = y*w+x;
                    visible[i] = 1;
                    //setPixel(f,&frameBuffer.color[ind*4]);
                }
            }
        
            i++;
            
            if(i>=MAX_RASTERIZED_PTS)
            {
                debugPrint("Reached limit of rasterizable points!\n");
                return i;
            }
            
            if(e >= 0)
            {
                y += iny;
                e-= dx;
            }
            e += dy; x += inx;
            
            
        }
    }
    else
    {
        dx <<= 1;
        e = dx - dy;
        dy <<= 1;
        while (iter)
        {
            if(y == y2)
                iter = false;
            
            t = ((float)(y-y1)*ddy)/(ddy*ddy);
            
            f.d = (float)z1 + ((float)z2 - z1)*t;
            
            X[i] = x;
            Y[i] = y;
            visible[i] = 0;
            
            if(x>=0 && x < w && y >= 0 && y < h )
            {
                if(!ztest || testFrag(f,x,y))
                {
                    visible[i] = 1;
                    //int ind = y*w+x;
                    //setPixel(f,&frameBuffer.color[ind*4]);
                }
            }
            
            i++;
            
            if(i>=MAX_RASTERIZED_PTS)
            {
                debugPrint("Reached limit of rasterizable points!\n");
                return i;
            }

            
            if(e >= 0)
            {
                x += inx;
                e -= dy;
            }
            e += dx; y += iny;
        }
    }
    
    return i;
}


void Rasterizer::rasterizeSegment( const Rasterizer::Vertex & va,
                                   const Rasterizer::Vertex & vb,
                                   bool ztest )

{
    int stride = frameBuffer.step/4;
    int h = frameBuffer.height;

    int n = getRasterizedSegmentPoints( rastX, rastY, rastVisible, va, vb, ztest );
    for( int i = 0; i < n; i++ )
    {
        if(rastVisible[i])
        {
            int ind = rastY[i]*stride+rastX[i];
            setPixel(state.color, &frameBuffer.color[ind*4]);
        }
    }
}
    
arma::mat Rasterizer::getRasterizedContourPoints( const Contour& ctr, bool ztest )
{
    arma::mat res;
    for( int i = 0; i < ctr.size()-1; i++ )
    {
        arma::mat v = getRasterizedSegmentPoints(ctr.points.col(i), ctr.points.col(i+1), ztest);
        if(v.n_cols)
            res = arma::join_horiz(res, v);
    }
    
    if( ctr.closed )
    {
        arma::mat v = getRasterizedSegmentPoints(ctr.last(), ctr.points.col(0), ztest);
        if(v.n_cols)
            res = arma::join_horiz(res, v);
    }
    
    return res;
}

void Rasterizer::rasterizeContour( const Contour& ctr, bool ztest )
{
    V3 normal = V3(0, 0, 1);
    
    for( int i = 0; i < ctr.size()-1; i++ )
    {
        rasterizeSegment(makeVertex(ctr.points.col(i), normal),
                         makeVertex(ctr.points.col(i+1), normal),
                         ztest);
    }
    
    if( ctr.closed )
    {
        rasterizeSegment(makeVertex(ctr.last(), normal),
                         makeVertex(ctr.points.col(0), normal),
                         ztest);
    }
}


void Rasterizer::clipAndRasterizeTriangle( const Rasterizer::Vertex & va,
                                     const Rasterizer::Vertex & vb,
                                     const Rasterizer::Vertex & vc )
    {
        ClippedTriangles clipped;
        clipTriangle(&clipped,va,vb,vc);
        for( int i = 0; i < clipped.nTriangles; i++ )
        {
            int tri = i*3;
            rasterizeTriangle(clipped.verts[tri],
                              clipped.verts[tri+1],
                              clipped.verts[tri+2]);
        }
    }
  
    
    

void Rasterizer::rasterizeTriangle( const Rasterizer::Vertex & va,
                                    const Rasterizer::Vertex & vb,
                                    const Rasterizer::Vertex & vc )
{
    int x, y, x0, x1, x2, y0, y1, y2, boxl, boxr, boxt, boxb, i;
    int f0, f1, f2, f3, f4, f5, f6, f7, f8;
    float ba, bb, bc, f9, f10, f11;
    unsigned char *scan, *ptr;
    Fragment A, B, C, v;
    
    // projected positions
    const V4 & a = va.projp;
    const V4 & b = vb.projp;
    const V4 & c = vc.projp;
    
    int *dscan, *dptr;
    
    if( state.cullCW && state.cullCCW )
        return;
    
    if( a.w<=0.0 || b.w<=0.0 || c.w<=0.0 )
        return;
    
    // culling
    f10 = (b.x - a.x)*(c.y - a.y);
    f11 = (b.y - a.y)*(c.x - a.x);
    
    if( ((f10<=f11)&&state.cullCW) || ((f10>=f11)&&state.cullCCW) )
        return;
    
    // tri vertices
    A.w = 1.0/a.w;
    A.d = DEPTH_MAX_HALF - (a.z*DEPTH_MAX_HALF*A.w);
    A.n = va.n;
    A.clr = va.clr*A.w;
    
    B.w = 1.0/b.w;
    B.d = DEPTH_MAX_HALF - (b.z*DEPTH_MAX_HALF*B.w);
    B.n = vb.n;
    B.clr = vb.clr*B.w;
    
    C.w = 1.0/c.w;
    C.d = DEPTH_MAX_HALF - (c.z*DEPTH_MAX_HALF*C.w);
    C.n = vc.n;
    C.clr = vc.clr*C.w;
    
    // convert to raster coordinates
    x0 = (1.0f + a.x*A.w) * 0.5f * frameBuffer.width;
    y0 = (1.0f + a.y*A.w) * 0.5f * frameBuffer.height;
    x1 = (1.0f + b.x*B.w) * 0.5f * frameBuffer.width;
    y1 = (1.0f + b.y*B.w) * 0.5f * frameBuffer.height;
    x2 = (1.0f + c.x*C.w) * 0.5f * frameBuffer.width;
    y2 = (1.0f + c.y*C.w) * 0.5f * frameBuffer.height;
    
    // compute bounding rectangle
    boxl = MIN3( x0, x1, x2 );
    boxr = MAX3( x0, x1, x2 );
    boxt = MIN3( y0, y1, y2 );
    boxb = MAX3( y0, y1, y2 );
    
    // clamp bounding rect to screen
    boxl =  boxl<0            ?             0  : boxl;
    boxr = (boxr>=frameBuffer.width)  ? (frameBuffer.width -1) : boxr;
    boxt =  boxt<0            ?             0  : boxt;
    boxb = (boxb>=frameBuffer.height) ? (frameBuffer.height-1) : boxb;
    
    // stop if the bounding rect is invalid or outside screen area
    if( boxl>=boxr || boxt>=boxb || boxb<0 || boxt>=frameBuffer.height || boxr<0 || boxl>=frameBuffer.width )
        return;
    
    // precompute factors for baricentric interpolation
    f0 = x1*y2 - x2*y1; f3 = y1-y2; f6 = x2-x1;
    f1 = x2*y0 - x0*y2; f4 = y2-y0; f7 = x0-x2;
    f2 = x0*y1 - x1*y0; f5 = y0-y1; f8 = x1-x0;
    
    f9  = 1.0 / (f3*x0 + f6*y0 + f0);
    f10 = 1.0 / (f4*x1 + f7*y1 + f1);
    f11 = 1.0 / (f5*x2 + f8*y2 + f2);
    
    int stride = frameBuffer.step/4;

    // iterate over scanlines in the bounding rectangle
    scan  = frameBuffer.color + (boxt*stride + boxl) * 4;
    dscan = frameBuffer.depth + (boxt*frameBuffer.width + boxl);
    
    for( y=boxt; y<=boxb; ++y, scan+=stride*4, dscan+=frameBuffer.width )
    {
        // iterate over pixels in current scanline
        for( dptr=dscan, ptr=scan, x=boxl; x<=boxr; ++x, ptr+=4, ++dptr )
        {
            // determine baricentric coordinates of current pixel
            ba = (f3*x + f6*y + f0) * f9;
            bb = (f4*x + f7*y + f1) * f10;
            bc = (f5*x + f8*y + f2) * f11;
            
            // skip invalid coordinates (outside of triangle)
            if( ba<0.0f || ba>1.0f || bb<0.0f || bb>1.0f || bc<0.0f || bc>1.0f )
                continue;
            
            // interpolate vertex coordinates and perform clipping
            v.d = A.d*ba + B.d*bb + C.d*bc;
            v.w = 1.0 / (A.w*ba + B.w*bb + C.w*bc);
            
            // interpolate vertex attributes
            v.clr.x = (A.clr.x*ba + B.clr.x*bb + C.clr.x*bc) * v.w;
            v.clr.y = (A.clr.y*ba + B.clr.y*bb + C.clr.y*bc) * v.w;
            v.clr.z = (A.clr.z*ba + B.clr.z*bb + C.clr.z*bc) * v.w;
            v.clr.w = (A.clr.w*ba + B.clr.w*bb + C.clr.w*bc) * v.w;
            v.clr = va.clr;
            
            if( v.w<=0 || v.d>DEPTH_MAX || v.d<0 )
                continue;
            
            drawPixel(v, ptr, dptr, state.depthTest);
        }
    }
}

void Rasterizer::drawPixel(const Rasterizer::Fragment & f, unsigned char * ptr, int * dptr, bool ztest )
{
    // depth test
    if( ztest && !(f.d <= *dptr) ) //!depth_fun( v->d, *dptr ) )
        return;
    
    ptr[RED  ] = f.clr.x*255.0;
    ptr[GREEN] = f.clr.y*255.0;
    ptr[BLUE ] = f.clr.z*255.0;
    ptr[ALPHA] = f.clr.w*255.0;
    
    if(ztest)
        *dptr = f.d;
}

void Rasterizer::setPixel(const V4 & clr, unsigned char * ptr )
{
    ptr[RED  ] = clr.x*255.0;
    ptr[GREEN] = clr.y*255.0;
    ptr[BLUE ] = clr.z*255.0;
    ptr[ALPHA] = clr.w*255.0;
}

void Rasterizer::draw(float x, float y, float w, float h)
{
    frameBuffer.image.draw(x,y,w,h);
}
 
static bool isVertexVisible( const Rasterizer::Vertex & v )
{
    const V4 & p = v.projp;
    return p.w+p.x > 0.0 &&
           p.w+p.y > 0.0 &&
           p.w+p.z > 0.0;
}

void Rasterizer::clipTriangle( Rasterizer::ClippedTriangles * clipped,
              const Rasterizer::Vertex & va,
              const Rasterizer::Vertex & vb,
              const Rasterizer::Vertex & vc)
{
    // avoid
    clipped->nTriangles = 0;
    
    // if triangle is totally hidden bail
    if(!isVertexVisible(va) &&
       !isVertexVisible(vb) &&
       !isVertexVisible(vc) )
        return;
    
    
}

    
}