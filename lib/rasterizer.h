//
//  swrasterizer.h
//  based on software rasterizer code by David Oberhollenzer
//
//  Created by colormotor on 5/29/14.
//  Copyright (c) 2014 colormotor. All rights reserved.
//

#pragma once
#include "colormotor.h"

namespace cm
{
    
class Rasterizer
{
public:
    Rasterizer();
    ~Rasterizer();
    
    void release();
    void init( int w, int h );
    
    void clear( float r, float g, float b, float a, bool depth=true);
    void clearDepth( float val = 1.0f );
    
    void setModelViewMatrix( const arma::mat& m );
    void setProjectionMatrix( const arma::mat& m );

    void setVertices( const arma::mat& V );
    void setNormals( const arma::mat& N );
    void setColors( const arma::mat& C );
    void setIndices( const std::vector<unsigned int>& I );

    void setFillMode(int fillMode);
    void setCullCCW(bool flag);
    void setCullCW(bool flag);
    void enableDepthBuffer(bool flag);
    void color( const arma::vec& clr );
    
    void rasterize();
    void draw( const cm::Mesh& mesh );
    void fill( const cm::Shape& shape );
    void fill( const cm::Contour& shape );
    
    void draw(float x=0, float y=0, float w = 0, float h=0);
    const Image & getImage() const { return frameBuffer.image; }
    
    struct Vertex
    {
        cm::V3 v;
        cm::V3 viewp;
        cm::V4 projp;
        cm::V3 n;
        cm::V4 clr;
    };
    
    struct Fragment
    {
        cm::V3 n;
        float w;
        int d;
        cm::V4 clr;
    };
    
    struct
    {
        int * depth;
        unsigned char * color;
        cm::Image image;
        int width;
        int height;
        int step;
    } frameBuffer;
    
    struct ClippedTriangles
    {
        Vertex verts[12]; // max 4 triangles resulting from clipping
        int nTriangles;
    };
    
    enum
    {
        DEPTH_MAX = 0x00FFFFFF,
        DEPTH_MAX_HALF = 0x007FFFFF,
        MAX_RASTERIZED_PTS = 4086
    };
    
    struct
    {
        bool cullCW;
        bool cullCCW;
        int fillMode;
        V4 color;
        float depthOffset;
        float slopeFactor;
        bool depthTest;
    } state;
    
    Rasterizer::Vertex makeVertex( const cm::V3& pos, const cm::V3& normal = cm::V3(1,0,0), const cm::V4& clr = cm::V4(1,1,1,1));
    
    arma::mat getRasterizedSegmentPoints( const arma::vec& a, const arma::vec& b, bool ztest=true );

    int getRasterizedSegmentPoints( int * X, int * Y, int * visible, const Rasterizer::Vertex & va,
                                    const Rasterizer::Vertex & vb,
                                    bool ztest = true );
    
    void rasterizeSegment( const Rasterizer::Vertex & va,
                          const Rasterizer::Vertex & vb,
                          bool ztest = true );
    
    arma::mat getRasterizedContourPoints( const cm::Contour& ctr, bool ztest=true );
    void rasterizeContour( const cm::Contour& ctr, bool ztest=true );

private:
    bool testFrag( const Rasterizer::Fragment & f, int x, int y );

    void rasterize( const std::vector<Rasterizer::Vertex> verts );
    
    void clipAndRasterizeTriangle( const Rasterizer::Vertex & va,
                           const Rasterizer::Vertex & vb,
                           const Rasterizer::Vertex & vc );
    
    void rasterizeTriangle( const Rasterizer::Vertex & va,
                            const Rasterizer::Vertex & vb,
                            const Rasterizer::Vertex & vc );
    
    
    void drawPixel(const Rasterizer::Fragment & f, unsigned char * ptr, int * dptr, bool ztest=true );
    void setPixel(const V4& clr, unsigned char * ptr );
    
    void clipTriangle( Rasterizer::ClippedTriangles * clipped,
                      const Rasterizer::Vertex & va,
                      const Rasterizer::Vertex & vb,
                      const Rasterizer::Vertex & vc);
    
    cm::M44 proj;
    cm::M44 modelView;
    cm::M44 normalMatrix;
    cm::M44 projFlipped;
    
    arma::mat normalBuffer;
    arma::mat vertexBuffer;
    arma::mat colorBuffer;
    
    std::vector<unsigned int> indexBuffer;
    int numVertices;
    int numIndices;
    
    int rastX[MAX_RASTERIZED_PTS];
    int rastY[MAX_RASTERIZED_PTS];
    int rastVisible[MAX_RASTERIZED_PTS];
};

}