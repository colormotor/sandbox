#pragma once
#include "colormotor.h"
namespace cm
{
	Shape trace( const Shape& s, const Image& img, float scalex=1., float scaley=1. );
	Shape trace( const Contour& s, const Image& img, float scalex=1., float scaley=1. );
}