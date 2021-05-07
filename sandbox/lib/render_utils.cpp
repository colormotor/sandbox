#include "render_utils.h"
using namespace arma;

namespace cm {
    
    float brushSize( float v, float minv, float maxv, float spread )
    {
        v = std::min(v, maxv);
        v = std::max(v, minv);
        v /= maxv;
        return (1.0 / (v*v+spread));
    }
    
    arma::vec brushSize( const arma::vec& v_, float minv, float maxv, float spread )
    {
        arma::vec v = arma::min(v_, arma::ones(v_.n_rows)*maxv);
        v = arma::max(v, arma::ones(v.n_rows)*minv);
        v /= maxv;
        return (1.0 / (v%v+spread));
    }

    void drawBrushExpSpeed( Image& img, const Contour & ctr_,  arma::vec S, int animIndex, float rMin, float rMax, float dist, float lowpass, float baseSpeed )
    {
        if(ctr_.size()<2)
            return;
        
        // Interpolate contour so we fill the trace
        float l = chordLength(ctr_.points, ctr_.closed);
        vec X = linspace(0,1,l/dist);
        //vec S = speed(ctr_.points, dt); //*interpRatio);
        //std::cout << arma::max(S) << std::endl;
        float vbar = arma::mean(S);
        S = exp(-(S+vbar) / (vbar+1e-100));

        S = interpolate(S, X);
        Contour ctr(interpolate(ctr_.points, X, ctr_.closed), ctr_.closed);
        
        // Make sure animation index is adjusted
        float interpRatio = (float)ctr.size() / ctr_.size();
        animIndex = interpRatio*animIndex;
        
        img.bind();
        int n = std::min(ctr.size(), animIndex);

        float w = 0.0;
        for( int i = 0; i < n; i++ )
        {
            const V2& p = ctr[i];
            float w2 = rMin + (rMax - rMin)*S[i];
            w += (w2 - w)*lowpass;
            img.draw( p.x-w, p.y-w, w*2, w*2 );
        }
        img.unbind();


    }


    void drawBrushExp( Image& img, const Contour & ctr_, int animIndex, float dt, float rMin, float rMax, float dist, float lowpass, float baseSpeed )
    {
        if(ctr_.size()<2)
            return;
        
        // Interpolate contour so we fill the trace
        float l = chordLength(ctr_.points, ctr_.closed);
        vec X = linspace(0,1,l/dist);
        vec S = speed(ctr_.points, dt); //*interpRatio);
        //std::cout << arma::max(S) << std::endl;
        float vbar = arma::mean(S);
        S = exp(-(S+vbar) / (vbar+1e-100));

        S = interpolate(S, X);
        Contour ctr(interpolate(ctr_.points, X, ctr_.closed), ctr_.closed);
        
        // Make sure animation index is adjusted
        float interpRatio = (float)ctr.size() / ctr_.size();
        animIndex = interpRatio*animIndex;
        
        img.bind();
        int n = std::min(ctr.size(), animIndex);

        float w = 0.0;
        for( int i = 0; i < n; i++ )
        {
            const V2& p = ctr[i];
            float w2 = rMin + (rMax - rMin)*S[i];
            w += (w2 - w)*lowpass;
            img.draw( p.x-w, p.y-w, w*2, w*2 );
        }
        img.unbind();

    }

    void drawBrush( Image& img, const Contour & ctr_, int animIndex, float dt, float brushSz, float brushMinv, float brushMaxv, float brushSpread, float dist  )
    {
        if(ctr_.size()<2)
            return;
        
        // Interpolate contour so we fill the trace
        float l = chordLength(ctr_.points, ctr_.closed);
        vec X = linspace(0,1,l/dist);
        vec S = speed(ctr_.points, dt); //*interpRatio);
        S = brushSize(S, brushMinv, brushMaxv, brushSpread );   
        S = interpolate(S, X);
        Contour ctr(interpolate(ctr_.points, X, ctr_.closed), ctr_.closed);
        
        // Make sure animation index is adjusted
        float interpRatio = (float)ctr.size() / ctr_.size();
        animIndex = interpRatio*animIndex;
        
        float w = brushSz * 0.1 * img.width();
        
        img.bind();
        int n = std::min(ctr.size(), animIndex);
        
        for( int i = 0; i < n; i++ )
        {
            const V2& p = ctr[i];
            float s = w*S[i]; //brushSize( S[i], brushMinv, brushMaxv, brushSpread );
            
            img.draw( p.x-s*0.5, p.y-s*0.5, s, s );
        }
        img.unbind();
    }

void drawBrushUniform( Image& img, const Contour & ctr_, int animIndex, float size, float dist )
    {
        if(ctr_.size()<2)
            return;
        
        // Interpolate contour so we fill the trace
        float l = chordLength(ctr_.points, ctr_.closed);
        vec X = linspace(0,1,l/dist);
        Contour ctr(interpolate(ctr_.points, X), ctr_.closed);
        
        // Make sure animation index is adjusted
        float interpRatio = (float)ctr.size() / ctr_.size();
        animIndex = interpRatio*animIndex;
        
        float w = size * img.width();
        
        img.bind();
        int n = std::min(ctr.size(), animIndex);
        
        for( int i = 0; i < n; i++ )
        {
            const V2& p = ctr[i];
            float s = w; 
            img.draw( p.x-s*0.5, p.y-s*0.5, s, s );
        }
        img.unbind();
    }

  /*
    void drawBrush( Image& img, const Contour & ctr_, int animIndex, float dt, float brushSz, float brushMinv, float brushMaxv, float brushSpread, float dist  )
    {
        if(ctr_.size()<2)
            return;
        
        // Interpolate contour so we fill the trace
        float l = chordLength(ctr_);
        vec X = linspace(0,1,l/dist);
        Contour ctr = interpolate(ctr_, X);
        
        // Make sure animation index is adjusted
        float interpRatio = (float)ctr.size() / ctr_.size();
        animIndex = interpRatio*animIndex;
        
        vec S = speed(ctr.points, dt*interpRatio);
        
        float w = brushSz * 0.1 * img.width();
        
        img.bind();
        int n = std::min(ctr.size(), animIndex);
        
        for( int i = 0; i < n; i++ )
        {
            const V2& p = ctr[i];
            float s = w*brushSize( S[i], brushMinv, brushMaxv, brushSpread );
            
            img.draw( p.x-s*0.5, p.y-s*0.5, s, s );
        }
        img.unbind();
    }
    */
    
    void drawGaussian2d( const arma::vec& mu, const arma::mat& Sigma, float radius, const arma::vec& clr )
    {
        mat V;
        vec d;
        eig_sym(d, V, Sigma.submat(span(0,1),span(0,1)));
        mat D = diagmat(sqrt(d));
        mat tm = eye(3,3);
        tm(span(0,1),span(0,1)) = V*D;
        tm(span(0,1),span(2,2)) = mu.subvec(0,1);
        
        gfx::pushMatrix(tm);
        V4 fillClr = clr;
        fillClr.w *= 0.3;
        
        gfx::color(fillClr);
        gfx::fillCircle(V2(0,0), radius, 100);
        gfx::color(clr);
        gfx::drawCircle(V2(0,0), radius, 100);
        gfx::popMatrix();
    }
    
    void drawGaussians2d( const arma::mat& Mu, const arma::cube& Sigma, float radius, const arma::vec& clr )
    {
        for( int i = 0; i < Mu.n_cols; i++ )
        {
            drawGaussian2d(Mu.col(i), Sigma.slice(i), radius, clr);
        }
    }
    
}
