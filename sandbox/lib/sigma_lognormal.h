#pragma once
#include "colormotor.h"

namespace cm
{
arma::mat slm_trajectory(  //arma::mat *Strokes_,
                             const arma::mat& Vp_,
                             const arma::vec& alpha,
                             const arma::vec& delta_t,
                             const arma::vec& Theta,
                             const arma::vec& d,
                           double dt,
                           // fitts law constants
                           double w=170.0,
                           double c=1.0,
                           double b=0.1 );
}