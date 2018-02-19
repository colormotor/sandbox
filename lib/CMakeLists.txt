cmake_minimum_required(VERSION 2.8.3)
project(pyautograff_utils)

set(CMAKE_MACOSX_RPATH 0)
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

add_definitions(-g)
# make it c++11 compatible
add_definitions(-std=c++11)
# Makes this cross compatible with colormotor bindings
add_definitions(-DSWIG_TYPE_TABLE=colormotor)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../)

set(src
	${PROJECT_SOURCE_DIR}/autograff_utils_wrap.cxx
	${PROJECT_SOURCE_DIR}/mpc.cpp
	${PROJECT_SOURCE_DIR}/sigma_lognormal.cpp
	${PROJECT_SOURCE_DIR}/render_utils.cpp
	${PROJECT_SOURCE_DIR}/tracer.cpp
	${PROJECT_SOURCE_DIR}/rasterizer.cpp
)

add_library(pyautograff_utils SHARED
	${src}
)

if(DEFINED ENV{COLORMOTOR_PATH})
	set(COLORMOTOR_PATH $ENV{COLORMOTOR_PATH})
else()
	message(FATAL_ERROR "You need to set the path for colormotor through the COLORMOTOR_PATH env var")
endif()

include_directories(
	  ${COLORMOTOR_PATH}/src/
	  ${COLORMOTOR_PATH}/addons/pycolormotor
	  ${PROJECT_SOURCE_DIR}/arpack-arma/include
)

if(APPLE)
	set(PYTHON_BASE_DIR $ENV{HOME}/miniconda2)
	message('${PROJECT_SOURCE_DIR}/../../../../colormotor/addons/pycolormotor/modules')

	include_directories(
		${PYTHON_BASE_DIR}/lib/python2.7/site-packages/numpy/core/include/
        ${PYTHON_BASE_DIR}/include
        ${PYTHON_BASE_DIR}/include/python2.7
	)

	link_directories(
		${PYTHON_BASE_DIR}/lib
	)
	#${PROJECT_SOURCE_DIR}/../../../../colormotor/addons/pycolormotor/modules
	target_link_libraries(pyautograff_utils
		${PYTHON_BASE_DIR}/lib/libopencv_core.dylib
		${PYTHON_BASE_DIR}/lib/libopencv_imgproc.dylib
		${PYTHON_BASE_DIR}/lib/libopencv_highgui.dylib
		${PYTHON_BASE_DIR}/lib/libopencv_imgcodecs.dylib
		${PYTHON_BASE_DIR}/lib/libarmadillo.dylib
		lapack
		blas
		glfw
		${COLORMOTOR_PATH}/addons/pycolormotor/modules/_cm.so
		${PYTHON_BASE_DIR}/lib/libpython2.7.dylib
		"-framework CoreFoundation"
		"-framework OpenGL"	
		"-framework Cocoa"
		"-framework AppKit"
		"-framework CoreVideo"
		"-framework QTKit"
		"-framework IOKit"
		"-framework AGL"
		"-framework GLUT"
		
	)

	add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND rm -f ${PROJECT_SOURCE_DIR}/../_autograff_utils.so 
    COMMAND ln -s ${PROJECT_SOURCE_DIR}/../libpyautograff_utils.dylib ${PROJECT_SOURCE_DIR}/../_autograff_utils.so 
	)
elseif(UNIX)
	find_package(Armadillo REQUIRED)
	find_package(PkgConfig REQUIRED)
	find_package(OpenCV REQUIRED)
	find_package(OpenGL REQUIRED)
	find_package(GLEW REQUIRED)
	pkg_check_modules(GTK2 REQUIRED gtk+-2.0)
	#pkg_search_module(PYTHON REQUIRED python>=2.6)
	# Nifty trick from http://stackoverflow.com/questions/24174394/cmake-is-not-able-to-find-python-libraries

	execute_process(COMMAND python -c "import sys; from distutils.sysconfig import get_python_inc; sys.stdout.write(get_python_inc())" OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS)

	execute_process(COMMAND python -c "import sys; import distutils.sysconfig as sysconfig; sys.stdout.write(sysconfig.get_config_var('LIBDIR'))" OUTPUT_VARIABLE PYTHON_LIBRARY_DIRS)

	execute_process(COMMAND python -c "import sys; import distutils.sysconfig as sysconfig; sys.stdout.write(sysconfig.get_config_var('LDLIBRARY'))" OUTPUT_VARIABLE PYTHON_LIBRARIES)

	message('Python lib dir')
	message(${PYTHON_LIBRARY_DIRS})
	message('Python lib')
	message(${PYTHON_LIBRARY_DIRS}/${PYTHON_LIBRARIES})
	
	include_directories(
		${GTK2_INCLUDE_DIRS}
		${ARMADILLO_INCLUDE_DIRS}
		${PYTHON_INCLUDE_DIRS}
		${PYTHON_LIBRARY_DIRS}//python2.7/dist-packages/numpy/core/include/
	)

	link_directories(
		${GTK2_LIBRARY_DIRS}
		${PYTHON_LIBRARY_DIRS}
	)

	add_definitions(
	${GTK2_CFLAGS_OTHER}
	)

	# Mess

	target_link_libraries(pyautograff_utils
	rt
	m
	dl
	Xrandr
	Xinerama
	Xxf86vm
	Xcursor
	Xext
	Xrender
	Xfixes
	X11
	pthread
	xcb
	Xau
	Xdmcp
	glfw
	${PYTHON_LIBRARY_DIRS}/libopencv_core.so
	${PYTHON_LIBRARY_DIRS}/libopencv_imgproc.so
	${PYTHON_LIBRARY_DIRS}/libopencv_legacy.so
	${PYTHON_LIBRARY_DIRS}/libopencv_highgui.so
	${PROJECT_SOURCE_DIR}/../../../../colormotor/addons/pycolormotor/modules/_cm.so
	${OPENGL_LIBRARIES}
	${GTK2_LIBRARIES}
	${ARMADILLO_LIBRARIES}
	${GLEW_LIBRARIES}
	${PYTHON_LIBRARY_DIRS}/${PYTHON_LIBRARIES}
	)

	add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND rm -f ${PROJECT_SOURCE_DIR}/../_autograff_utils.so 
    COMMAND ln -s ${PROJECT_SOURCE_DIR}/../libpyautograff_utils.so ${PROJECT_SOURCE_DIR}/../_autograff_utils.so 
	)
endif()










