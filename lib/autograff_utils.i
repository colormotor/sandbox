//(directors="1")
%module autograff_utils
#pragma SWIG nowarn=322,362

%include <typemaps.i>
%include <stl.i>
%include <std_string.i>
%include <std_vector.i>

%{
// Fuckin idiotic, mac defines "check"
#ifdef check
	#undef check
#endif
#define SWIG_FILE_WITH_INIT
#include "mpc.h"
#include "sigma_lognormal.h"
#include "render_utils.h"
#include "rasterizer.h"
#include "tracer.h"
#include "Python.h"
%} 

%include "armanpy.i"
%import "cm.i"

using namespace std;

%include "mpc.h"
%include "sigma_lognormal.h"
%include "render_utils.h"
%include "rasterizer.h"
%include "tracer.h"
