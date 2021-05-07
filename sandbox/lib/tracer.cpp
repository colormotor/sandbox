/*
 *  Tracer.cpp
 *
 *  Created by Daniel Berio on 4/29/13.
 *  http://www.enist.org
 *  Copyright 2013. All rights reserved.
 *
 */


#include "tracer.h"

namespace cm
{
	V2 closestPoint(const V2& a, const V2& b, const V2& p)
	{
	    V2 ap = p - a;
	    V2 ab = b - a;
	    float ab2 = ab.x*ab.x + ab.y*ab.y;
	    float ap_ab = ap.x*ab.x + ap.y*ap.y;
	    float t = ap_ab / ab2;
	    
	    if (t < 0.0f) t = 0.0f;
	    else if (t > 1.0f) t = 1.0f;
	    
	    return a + ab * t;
	}

    /*
     m=(y2-y1)/(float)(x2-x1);
     float y=y1;
     int x;
     for (x=x1;x<=x2;x++,y+=m)
     putpixel(x,y,color);
    */
    
    
    static int traceLine(Shape* shape, const Image& img, const V2& a, const V2& b, float scalex, float scaley, int on )
	{
		float x0 = a.x*scalex;
		float y0 = a.y*scaley;
		float x1 = b.x*scalex;
		float y1 = b.y*scaley;
		float x = x0;
		float y = y0;
        
        V2 d = b-a;
        float len = arma::norm(d);
        d/=len;

        int w = img.width();
        int h = img.height();
        int step = img.step();
        int chans = img.mat.channels();
        int stride = step / chans;

        // Add an initial contour if shape is empty
		if(!shape->size())
			shape->add(Contour());
		
#define THRESH 155
#define FUZZY_DIF(a,b) (abs(b-a)>THRESH)

#define CLOSEST(x,y) V2(x,y) //closestPoint(a,b,V2(x,y))
		float prevx = x;
		float prevy = y;
		float dist = 0.0;
        
        
		int v = 0;
		for(;;){
			Contour & path = shape->last();
			
			int iy = roundf(y);
			int ix = roundf(x);
			if(ix >= w || 
			   ix < 0  ||
			   iy >= h ||
			   iy < 0 )
				v = 0;
			else 
				v = img.mat.data[iy*step+(ix*chans)];
			
            
			if(v)
			{
				if(on)
				{
					path.addPoint(CLOSEST(prevx,prevy));
					shape->add(Contour());
				}
				on = 0;
			}
			else // on
			{
				if(!on)
				{
					path.addPoint(CLOSEST(x,y));
				}
				
				on = 1;
			}
			/*
			if( FUZZY_DIF(v, prevv) ) // v!=prevValue )
			{
				if( v > THRESH )
				{
					// point is occluded, add end point
					path.addPoint(CLOSEST(prevx,prevy));
					shape->add(Contour());
				}
				else
				{
					// start a new contour and add point
					path.addPoint(CLOSEST(prevx,prevy));
				}
			}
			else
			{
				if( feq(x,x0) && feq(y,y0) && v < THRESH)
					// first point in line
					path.addPoint(CLOSEST(prevx,prevy));
			}
            
            prevv = v;*/
			prevx = x;
			prevy = y;
			
			if (dist>=len) break;
            x+=d.x;
            y+=d.y;
            dist+=1.0;
		}
		
		if(on)
		{
			shape->last().addPoint(b); // last point
		}
		
		return on;
	}
    
    /*
	int trace_line(Shape * shape, const Image & img, const V2 & a, const V2 & b, float scalex, float scaley, int prevValue )
	{
		int x0 = a.x*scalex;
		int y0 = a.y*scaley;
		int x1 = b.x*scalex;
		int y1 = b.y*scaley;
		int x = x0;
		int y = y0;
		
		int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
		int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1; 
		int err = (dx>dy ? dx : -dy)/2, e2;
		
		// Add an initial contour if shape is empty
		if(!shape->size())
			shape->appendContour();
		
#define THRESH 5
#define FUZZY_DIF(a,b) (abs(b-a)>5)
#define CLOSEST(x,y) closestPoint(a,b,V2(x,y))
		int prevx = x;
		int prevy = y;
		
		for(;;){
			Contour & path = shape->last();
			
			int v = img.getIntensity(x,y)*255;
			
			if( FUZZY_DIF(v,prevValue) ) //v!=prevValue )
			{
				if( v > THRESH )
				{
					// point is occluded, add end point
					path.addPoint(CLOSEST(prevx,prevy));
					shape->appendContour();
				}
				else
				{
					// start a new contour and add point
					path.addPoint(CLOSEST(prevx,prevy));
				}
			}
			else 
			{
				if( x==x0 && y==y0 && v < THRESH)
					// first point in line 
					path.addPoint(CLOSEST(prevx,prevy));
			}
			prevValue = v;
			prevx = x;
			prevy = y;
			
			if (x==x1 && y==y1) break;
			e2 = err;
			if (e2 >-dx) { err -= dy; x += sx; }
			if (e2 < dy) { err += dx; y += sy; }
		}
		
		
		return prevValue;
	}*/
	
	
	Shape trace( const Contour& path, const Image& img, float scalex, float scaley )
	{
		Shape tmp;
		
		if(path.size() < 2)
			return tmp;
		int n = path.size()-1;
		if( path.closed )
			n+=1;
		int v = 0;
		for( int j = 0; j < n; j++ )
		{
			const V2 & a = path.getPoint(j);
			const V2 & b = path.getPoint((j+1)%path.size());
			v = traceLine(&tmp, img, a, b, scalex, scaley, v);
		}
		if( v < THRESH && path.closed && tmp.size())
		{
			Contour & last = tmp.last();
			if(path.size())
				last.addPoint(path.getPoint(0));
		}
	
		// check that shape is valid
		Shape res;
		for( int i = 0; i < tmp.size(); i++ )
			if(tmp[i].size()>1)
				res.add(tmp[i]);
		
		return res;
	}
	
	Shape trace( const Shape& s, const Image& img, float scalex, float scaley )
	{
		Shape res;
		for( int i = 0; i < s.size(); i++ )
			res.add(trace(s[i], img, scalex, scaley));
		return res;
	}
					   
	
}