//
//  sigma_lognormal.cpp
//  gmr_test
//
//  Created by Daniel Berio on 26/04/16.
//  Copyright (c) 2016 Daniel Berio. All rights reserved.
//

#include "sigma_lognormal.h"

using namespace arma;

namespace cm
{
double fitts( double d,
              double w,
              double c,
              double b)
{
    return b*log2((d*2)/w + c);
}

double lognormal( double x, double x0, double mu, double sigma )
{
    double eps = 1e-8;
    x = x - x0;
    x = std::max(x, eps);
    
    return exp( -pow( log(x) - mu ,2) / (2*sigma*sigma) ) / (x*sqrt(2.0*PI)*sigma);
}


// void slm_mu_sigma( vec* mu, vec* sigma, const vec& alpha, const vec& d )
// {
//     *sigma = log(1.0 + 1E-10 + alpha);
//     *mu = -log(  -(exp(-3* (*sigma)) - exp(3* (*sigma))) % (1.0 / d) );
// }

void slm_mu_sigma( vec* mu, vec* sigma, const vec& Ac, const vec& T )
{
    *sigma = sqrt(- log(1.0 - Ac) );
    *mu = 3. * (*sigma) - log( (-1. + exp(6.*(*sigma))) / T );
//    return mu, sigma

  //  *sigma = log(1.0 + 1E-10 + alpha);
   // *mu = -log(  -(exp(-3* (*sigma)) - exp(3* (*sigma))) % (1.0 / d) );
}


double lognormal_interpolate( double a,
                              double b,
                              double x,
                              double x0,
                              double mu,
                              double sigma )
{
    x = x - x0;
    x = std::max(x, 1E-200);
    return a + (b - a)*0.5*(1 + erf( (log(x) - mu) / (sqrt(2)*sigma) ));
}

    /// Returns keypoints of a longormal trajectory, given a vector of (unscaled) strokes'''
uvec slm_keypoints( const mat& Strokes )
{
    uvec I({0}); // first point
    unsigned int m = Strokes.n_rows;
    unsigned int n = Strokes.n_cols;
    
    for( unsigned int i = 0; i < m-1; i++ )
    {
        vec r1=Strokes.row(i).t();
        vec r2=Strokes.row(i+1).t();
        uvec inds = find(diff(r2 > r1) != 0);
        uword amax;
        r2(inds).max(amax);
        unsigned int imax = inds[amax];
        I = join_vert(I, uvec({imax}));
    }
    
    I = join_vert(I, uvec({n-1})); // last
    return I;
}

/// Explicit virtual target reparametrisation of a sigma-lognormal trajectory
/// using alpha ]0,1] as an assymetry parameter, and d for the duration of the lognormal
mat slm_trajectory(  //mat *Strokes_,
                     const mat& Vp_,
                     const vec& alpha,
                     const vec& delta_t_,
                     const vec& Theta,
                     const vec& d_,
                     double dt,
                     // fitts law constants
                     double w,
                     double c,
                     double b )
{
    mat Vp = Vp_;
    int m = Vp.n_cols-1;
    int dim = Vp.n_rows;

    if(dim == 2)
        Vp = join_vert(Vp, zeros(1,m+1));
    
    // copy for mod
    vec delta_t = delta_t_;
    vec d = d_;
    
    vec mu, sigma;
    slm_mu_sigma(&mu, &sigma, alpha, d);

    // compute t0 values from delta
    vec T0 = zeros(m)+0.0;
    if(m > 1)
    {
        for( int i = 1; i < m; i++ )
            T0[i] = delta_t[i] * d[i]; 
        //T0.subvec(1, T0.size()-1) = delta_t.subvec(1, delta_t.size()-1) * d.subvec(1, d.size()-1);
        T0 = cumsum(T0);
    }
    vec T1 = T0;

    // offsets
    mat dP = diff(Vp, 1, 1);
    vec D = sqrt(sum(dP % dP, 0)).t();
    
    // for( int i=0; i < m; i++ )
    // {
    //     double f = fitts(D[i], w, c, b);
    //     d[i] = d[i] + f;
    //     if(i < m-1)
    //         delta_t[i+1] = delta_t[i+1] + f;
    // }

    // Add onsets in order to shift lognormal to start
    T0 = T0 - exp(mu - sigma*3);
    
    // endtime
    double endt = T0[m-1] + exp( mu[m-1] + sigma[m-1]*3 );

    // time steps
    vec T = cm::regspace(0.0, dt, endt);

    int n = T.size();
    cube Ps = zeros(m, 3, n);
    mat Strokes = zeros(m,n);

    double theta, s, theta_1, theta_2, Lambda;
    for( int i = 0; i < m; i++ )
    {
        theta = Theta[i];
        
        s = 1;
        if( fabs(sin(theta)) > 0.0 )
            s = (theta*2) / (2*sin(theta));
        
        theta_1 = theta; theta_2 = -theta;
        
        for( int j = 0; j < n; j++ )
        {
            theta = lognormal_interpolate(theta_1, theta_2, T[j], T0[i], mu[i], sigma[i]);
            mat A({{cos(theta)*s, -sin(theta)*s, 0},
                   {sin(theta)*s, cos(theta)*s, 0},
                   {0, 0, 1}} );
            
            Lambda = lognormal(T[j], T0[i], mu[i], sigma[i]);
            Strokes(i,j) = norm(dP.col(i))*Lambda;
            Ps.subcube(i, 0, j, i, 2, j)  = A*dP.col(i)*Lambda*dt;
        }
    }
    
    mat V = sum(Ps, 0);
    mat P = cumsum(V, 1) + repmat(Vp.col(0), 1,n);
    
    //if(Strokes_)
    //    *Strokes_ = Strokes;
    
    if(dim == 2)
        P = P.rows(0,1);
    
    return P;
}


}