//  Model Predictive Control trajectory generation
//  Created by Daniel Berio on 26/04/16.
//  Copyright (c) 2016 Daniel Berio. All rights reserved.
//

#include "mpc.h"
#include "render_utils.h"
//#include <LinAlg/LapackWrapperExtra.h>
//#include <SymEigsSolver.h>

// We are going to calculate the eigenvalues of M

using namespace arma;
using namespace cm;

namespace cm
{

// void symEigs( arma::vec* Dv, arma::mat* V, arma::mat A, int numEigs, int convFactor)
// {
//     // Construct matrix operation object using the wrapper class DenseGenMatProd
//         DenseGenMatProd<double> op(A);
//         // Construct eigen solver object, requesting the largest three eigenvalues
//         SymEigsSolver< double, LARGEST_ALGE, DenseGenMatProd<double> > eigs(&op, numEigs, convFactor);
        
//         eigs.init();
//         int nconv = eigs.compute();
        
//         if(nconv > 0)
//         {
//             *Dv = eigs.eigenvalues();
//             *V = eigs.eigenvectors(nconv);
//         }
//         else
//         {
//             *Dv = arma::zeros(2);
//             *V = arma::eye(2,2);
//         }
//     //assert(0);
// }


double factorials[15] =
{
   1,
    1,
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    39916800,
    479001600,
    6227020800,
    87178291200,
};

static double cmd_r(int order, double period)
{
    return (double)factorials[order+1] / pow(period, order+1);
}
    
static double gain(int order, double period)
{
    
    complexd tf = (1. / std::pow(complexd(0., (2.*PI) / period ), order));
    return std::abs(tf);
}

static double getExtent( const std::vector<arma::vec>& Mu )
{
    Box box;
    for( int i = 0; i < Mu.size(); i++ )
        box.includeAt( i, Mu[i] );
    // diagonal length
    return arma::norm(box.min() - box.max());
}

static mat transformSigma( const mat& m, const mat& Sigma)
{
    mat V;
    vec d;
    eig_sym(d, V, Sigma);
    mat D = diagmat(d);
    V = m*V;
    return V*D*inv(V);
}
    
static mat makeSigma( float rot, vec scale )
{
    mat Q = rot2d(rot, false);
    mat Lambda = diagmat(scale%scale);
    return Q * Lambda * inv(Q);
}


static void discretizeSystem(arma::mat* _A, arma::mat* _B, const arma::mat& A, const arma::mat& B, double dt, bool zoh=true  )
{
    if(zoh)
    {
        // matrix eponential trick
        // From Linear Systems: A State Variable Approach with Numerical Implementation pg 215
        // adapted from scipy impl

        mat EM = join_horiz(A, B);
        mat zero = join_horiz(zeros(B.n_cols, B.n_rows), 
                              zeros(B.n_cols, B.n_cols));
        EM = join_vert(EM, zero);
        mat M = expmat(EM * dt);
        *_A = M.submat(span(0, A.n_rows-1), span(0, A.n_cols-1));
        *_B = M.submat(span(0, B.n_rows-1), span(A.n_cols, M.n_cols-1));
    }
    else  // euler
    {
        *_A = eye(A.n_rows, A.n_cols) + A*dt;
        *_B = B * dt;
    }
}
    
    
    
/// Creates random gaussians as targets for LQR constraints
void MPC::makeRandomGaussians(const arma::mat& V, const arma::vec& covscale, bool semiTied, double theta, double minRnd )
{
    int m = V.n_cols;
    
    Mu = V;
    Sigma = zeros(2,2,m);
    vec s;
    
    for( int i = 0; i < m; i++ )
    {
        vec s = (minRnd+(randu(2)*(1.-minRnd))) % covscale;
        if(!semiTied)
            theta = frand(-PI, PI);
        mat sigm = makeSigma(theta, s);
        Sigma.slice(i) = makeSigma(theta, s);
    }
}


/// Prepares the state sequence for LQR computation with n time steps
/// Also automatically sets up a chain of integrators
void MPC::init( int n )
{
    int m = Mu.n_cols;
    assert(m);
    int muDim = Mu.n_rows;
    int cDim = order*dim;
    
    // Scaling factor
    float scale = globScale; 
    float scale2 =   scale*scale; 

    // equally distributed optimal states
    // Could try something along the lines of Fitts here based on covariance.
    //std::vector<int> qList(n);
    arma::ivec qList = arma::conv_to<arma::ivec>::from(
            arma::linspace(0, (float)m-0.1, n) );
        
    // int inc = n / m;
    // for( int i = 0; i < n; i++ )
    //     qList[i] = floorf((float)i / (inc+1));
    
    // precision matrices
    mat Lambda = zeros(muDim, m*muDim);
    for( int i = 0; i < m; i++ )
    {
        if(false) //viaPoints)
            Lambda.cols(i*muDim, muDim*(i+1)-1) = eye(muDim, muDim);
        else
            Lambda.cols(i*muDim, muDim*(i+1)-1) = inv(Sigma.slice(i)*scale2);//;
    }
    Q = zeros(cDim*n, cDim*n); // Precision matrix
    MuQ = zeros(cDim, n); // Mean vector
    
    if(viaPoints)
    {
        for( int state=0; state < m; state++ )
        {
          int i = (int)(0. + ((float)(n-1)/(m-1)) * state);
          Q.submat(i*cDim,
                   i*cDim,
                   i*cDim+muDim-1,
                   i*cDim+muDim-1) =
            Lambda.cols(state*muDim, (state+1)*muDim-1);
          MuQ.submat(span(0,muDim-1), span(i,i)) = Mu.col(state)*scale;
        }
    }
  /*
        for( int i = 0; i<qList.n_rows; i++ )
        {
            // Precision matrix based on state sequence
            if(qList[i] != prevState || i == qList.n_rows-1)
            {
                //prevState = qList[i];
                Q.submat(i*cDim,
                         i*cDim,
                         i*cDim+muDim-1,
                         i*cDim+muDim-1) =
                Lambda.cols(prevState*dim, (prevState+1)*muDim-1);
            }
            else
            {
                Q.submat(i*cDim,
                         i*cDim,
                         i*cDim+muDim-1,
                         i*cDim+muDim-1) = zeros(dim, dim);
            }
            prevState = qList[i];
            // Mean vector
            MuQ.submat(span(0,dim-1), span(i,i)) = Mu.col(qList[i])*scale;
        }
        }*/
    else
    {
      arma::ivec I = arma::conv_to<arma::ivec>::from(
                                      arma::linspace(0, (float)m-0.1, n) );
        for( int i = 0; i<n; i++ )
        {
            // Precision matrix based on state sequence
            Q.submat(i*cDim,
                     i*cDim,
                     i*cDim+muDim-1,
                     i*cDim+muDim-1) =
            Lambda.cols(I[i]*muDim, (I[i]+1)*muDim-1);
            // Mean vector
            MuQ.submat(span(0, muDim-1), span(i,i)) = Mu.col(I[i])*scale;
        }
    }
    
    if(endWeight > 0.0)
    {
        // Set last value for Q to enforce 0 boundary condition
        int ind = qList.n_rows-1;
        for( int i = muDim; i < cDim; i++ )
            Q(ind*cDim+i, ind*cDim+i) = endWeight; //pow(100.0, order); //  / pow(0.1, i/2 + 1);
    
    }

    if(startWeight > 0.0)
    {
        // Set first value for Q to enforce 0 boundary condition
        int ind = 0;
        for( int i = muDim; i < cDim; i++ )
            Q(ind*cDim+i, ind*cDim+i) = startWeight; //pow(100.0, order); //  / pow(0.1, i/2 + 1);
    }
    
    x0 = Mu.col(0);
}

/// Inits the system matrices with a chain of integrators
void MPC::initIntegratorChain()
{
    A = zeros(order, order);
    A.submat(0, 1, order-2, order-1)
    = eye(order-1, order-1);
    B = join_vert(zeros(order-1,1),
                      ones(1,1));
}

// Compute r based on a simple harmonic motion and maximum displacement d
void MPC::SHM_r(double d, double duration)
{
    int m = Mu.n_cols;
    assert(m);
    double period = (duration/(m));
    double omega = (2. * PI) / period;
    r = 1. / pow(d * pow(omega,order), 2);
}

/// Computes iterative finite horizon LQR solution
void MPC::computeIterative()
{
    // pre scale to avoid numerical issues
    float scale = globScale; //(1. / getExtent(Mu))*globScale;
    float scale2 =   scale*scale; // for covariance
    
    arma::mat res;
    int n = MuQ.n_cols;
    int cDim = dim*order;
    int muDim = Mu.n_rows;

    mat A, B, C, R, Su, Sx;
    
    discretizeSystem(&A, &B, this->A, this->B, dt, zoh);
    
    // make multivariate
    A = kron(A, eye(dim, dim));
    B = kron( B, eye(dim, dim) );
    
    // Sensor matrix (assuming only position is observed)
    C = kron( join_horiz(ones(1,1), zeros(1, order-1)),
             eye(dim, dim));
    
    R =  eye(dim, dim) * r;
    
    cube P = zeros(cDim, cDim, n);
    int i = n-1;
    P.slice(i) = Q.submat(span(i*cDim,i*cDim+cDim-1),span(i*cDim,i*cDim+cDim-1));
    
    mat d = zeros(cDim, n);
    // Riccati recursion
    for( int i = n-2; i >=0; i-- )
    {
        mat Qi = Q.submat(span(i*cDim,i*cDim+cDim-1),span(i*cDim,i*cDim+cDim-1));
        mat P2 = P.slice(i+1);
        P.slice(i) = Qi - (A.t() * (P2 * B * inv(B.t() * P2 * B + R) * B.t() * P2 - P2 ) * A);
        d.col(i) = (A.t() - A.t() * P2 * B * inv(B.t() * P2 * B + R) * B.t())
                   * (P2 * (A * MuQ.col(i) -  MuQ.col(i+1)) + d.col(i+1));
    }
    
    // Initial condition given by first mean, and zero derivatives
    vec x0 = join_vert(this->x0*scale, zeros(cDim-muDim));
    
    mat Y = zeros(cDim, n);
    vec x = x0;
    
    for( int i = 0; i < n; i++ )
    {
        Y.col(i) = x;
        mat p = P.slice(i);
        vec mu = MuQ.col(i);
            
        mat G = inv( B.t() * p * B + R ) * B.t();
            
        // feedback gain
        mat K = G * p * A;
        // Feedforward term
        mat M = -G * (p * (A * mu - mu) + d.col(i));
        // Control command (highest order derivative of system)
        vec u = K * (mu - x) + M;
        //  New state
        x = A*x + B*u;
    }
    
    this->P = Y.submat(0,0, dim-1, n-1) / scale; // unscale
}

/// Computes batch LQR solution (MPC)
void MPC::computeBatch()
{
    // pre scale to avoid numerical issues
    float scale = globScale; //(1. / getExtent(Mu))*globScale;
    float scale2 =   scale*scale; // for covariance
    
    arma::mat res;
    int n = MuQ.n_cols;
    //int dim = MuQ.n_rows / A.n_rows;
    int cDim = dim*order;
    int muDim = Mu.n_rows;

    mat A, B, C, R, Su, Sx;
    
    discretizeSystem(&A, &B, this->A, this->B, dt, zoh);
    
    // make multivariate
    A = kron(A, eye(dim, dim));
    B = kron( B, eye(dim, dim) );

    // Sensor matrix (assuming only position is observed)
    C = kron( join_horiz(ones(1,1), zeros(1, order-1)),
             eye(dim, dim));
    
    R = kron( eye(n-1, n-1),
             eye(dim, dim) * r );
    
    // Sx and Su matrices for batch LQR
    Su = zeros(cDim*n, dim*(n-1));
    Sx = kron( ones(n,1),
              eye(dim*order, dim*order));
    mat M = B;
    for( int i = 1; i < n; i++ )
    {
        Sx.rows( i*cDim, n*cDim-1 ) =
        Sx.rows( i*cDim, n*cDim-1 ) * A;
        Su.submat( i*cDim, 0,
                  (i+1)*cDim-1, i*dim-1 ) = M;
        M = join_horiz( A*M.cols(0, dim-1), M );
    }
    
    // Initial condition given by first mean, and zero derivatives
    arma::vec x0 = join_vert(this->x0*scale, zeros(cDim-muDim));
    // Flatten Mu's
    mat Xi_hat = reshape(MuQ, cDim*n, 1);
    
    mat SuInvSigmaQ = Su.t()*Q;
    mat Rq = SuInvSigmaQ*Su + R;
    mat rq = SuInvSigmaQ * (Xi_hat - Sx*x0);

    // get covariance
    if(computeCovariance)
    {
       //Cov = (Su * inv(Rq + 1e-7*eye(Rq.n_rows,Rq.n_cols)) * Su.t()) / (scale2);
       Cov = (Su * inv(Rq) * Su.t()) / (scale2);
       mse = arma::abs( Xi_hat.t() * Q * Xi_hat - rq.t() * arma::inv(Rq) * rq )[0] / (dim*n);
    }
    
    // least squares solution
    vec u = solve(Rq,rq);

    mat Y = reshape( Sx*x0 + Su*u, cDim, n );
    this->P = Y.submat(0,0, dim-1, n-1) / scale; // unscale
}

/// Draws the gaussian components
void MPC::drawGaussians( float radius, const arma::vec& clr)
{
    drawGaussians2d(Mu, Sigma, radius, clr);
}

void MPC::stochasticSample(int numSamples, int numEigs, int convFactor, float sig)
{
    stochasticSamples.clear();

    arma::vec Dv;
    arma::mat V;
    /*
    symEigs(&Dv, &V, Cov, numEigs, convFactor);
    mat D = arma::diagmat(arma::sqrt(Dv));
    
    int n = P.n_cols;
    for( int i = 0; i < numSamples; i++ )
    {
        float sigma = sqrt(mse)*sig;
        vec o = randn(V.n_cols) * sigma;
        mat offset = V * D * o;
        mat O = reshape( offset, this->A.n_cols*2, n );  // 2 is a big hack!!!
        stochasticSamples.push_back(this->P + O.submat(0,0, 2-1, n-1));
    } */
}

}
