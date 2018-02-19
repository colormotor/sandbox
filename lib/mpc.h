#pragma once
#include "colormotor.h"

namespace cm
{
    // TODO implement discretization: Euler and ZOH?
    class MPC
    {
    public:
        MPC(){}
        ~MPC(){}
        
        // Call in the following order
        
        /// Creates random gaussians as targets for LQR constraints
        void makeRandomGaussians(const arma::mat& V, const arma::vec& covscale, bool semiTied=false, double theta=0., double minRnd=0.2 );
        
        /// Prepares the state sequence for LQR computation with n time steps
        /// Also automatically sets up a chain of integrators, ands sets initial condition to the first mean.
        void init( int n );
        
        /// Inits the system matrices with a chain of integrators
        void initIntegratorChain();

        // Compute r based on a simple harmonic motion and maximum displacement d
        void SHM_r(double d, double duration);
        
        /// Computes iterative finite horizon LQR solution
        void computeIterative();
        /// Computes batch LQR solution (MPC)
        void computeBatch();
        
        /// Draws the gaussian components
        void drawGaussians( float radius, const arma::vec& clr=arma::vec({0,0.5,1.0,0.4}) );
        
        void stochasticSample(int numSamples, int numEigs, int convFactor, float sigma);
        int getNumStochasticSamples() const { return stochasticSamples.size(); }
        arma::mat getStochasticSample(int i) { return stochasticSamples[i]; }
        
        // Dynamic matrices (1d continuous)
        arma::mat A;
        arma::mat B;
        
        // State sequences
        arma::mat MuQ;
        arma::mat Q;
        
        arma::vec x0;
        arma::mat Mu;
        arma::cube Sigma;
        
        int order=4;
        int dim=2;
        float dt=0.01;
        float r=0.01;

        float globScale=0.01; // Global scaling factor (improves batch performance)

        // Lqr states input data
        float endWeight=1000000;
        float startWeight=1000000;
        bool viaPoints=false;
        
        bool zoh=true; // Use zero order hold approx
        bool computeCovariance=false;
        float mse=0.0;
        
        // Outputs
        arma::mat Cov;
        arma::mat P;

        std::vector<arma::mat> stochasticSamples;
    };
}
