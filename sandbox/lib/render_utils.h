#pragma once
#include "colormotor.h"
namespace cm
{
    float brushSize( float v, float minv, float maxv, float spread = 3.0 );
    void drawBrush( Image& img, const Contour & ctr_, int animIndex, float dt, float brushSz, float brushMinv, float brushMaxv, float brushSpread, float dist=1.  );
    void drawBrushExp( Image& img, const Contour & ctr_, int animIndex, float dt, float rMin, float rMax, float dist=1., float lowpass=1., float baseSpeed=0.  );
    void drawBrushExpSpeed( Image& img, const Contour & ctr_,  arma::vec S, int animIndex, float rMin, float rMax, float dist=1., float lowpass=1., float baseSpeed=0. );
    void drawBrushUniform( Image& img, const Contour & ctr_, int animIndex, float size, float dist=1. );

    /// Draws a 2d Gaussian
    void drawGaussian2d( const arma::vec& mu, const arma::mat& Sigma, float radius=1., const arma::vec& clr=arma::vec({0,0.5,1.0,0.4}) );
    /// Draws a set of 2d Gaussians
    void drawGaussians2d( const arma::mat& Mu, const arma::cube& Sigma, float radius=1., const arma::vec& clr=arma::vec({0,0.5,1.0,0.4}) );
    
}
