cmake_minimum_required(VERSION 2.8.3)
project(pyautograff_utils)

set(CMAKE_MACOSX_RPATH 0)
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

add_definitions(-g)
# make it c++11 compatible
# add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Makes this cross compatible with colormotor bindings
add_definitions(-DSWIG_TYPE_TABLE=colormotor)
add_definitions(-DAG_DISABLE_COLORMOTOR)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../)

find_package(PythonLibs REQUIRED)


set(src
	${PROJECT_SOURCE_DIR}/autograff_utils_wrap.cxx
	${PROJECT_SOURCE_DIR}/mpc.cpp
	${PROJECT_SOURCE_DIR}/sigma_lognormal.cpp
	${PROJECT_SOURCE_DIR}/render_utils.cpp
	${PROJECT_SOURCE_DIR}/tracer.cpp
	${PROJECT_SOURCE_DIR}/rasterizer.cpp
)

# if(DEFINED ENV{COLORMOTOR_PATH})
# 	set(COLORMOTOR_PATH $ENV{COLORMOTOR_PATH})
# else()
# 	message(FATAL_ERROR "You need to set the path for colormotor through the COLORMOTOR_PATH env var")
# endif()

set(COLORMOTOR_PATH "${PROJECT_SOURCE_DIR}/../../../../../colormotor")

include_directories(
	  ${COLORMOTOR_PATH}/src/
	  ${COLORMOTOR_PATH}/addons/pycolormotor
	  ${PROJECT_SOURCE_DIR}/arpack-arma/include
)

if(APPLE)
	# Nasty workaround to avoid conflicts (which popped out of nowhere, but make sense)
	# with multiple linking of Python.:
	# https://stackoverflow.com/questions/25421479/clang-and-undefined-symbols-when-building-a-library
	# The crpytic PyThreadState_Get() error message, means that 
	# the function is defined BOTH in the py executable and in the linked lib, evidently the old Python
	# interpreter used to link to a shared lib while the new one has all static linked?. 
	# Ideally, we would want to use distutils to get information on how to link python, quite involved anyhow
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
	endif()

	set(BASE_SEARCH_DIR $ENV{HOME}/opt/miniconda3)

	set(CMAKE_EXE_LINKER_FLAGS "-rpath ${BASE_SEARCH_DIR}/lib")


	include_directories(
		${BASE_SEARCH_DIR}/lib/python3.8/site-packages/numpy/core/include/
		${BASE_SEARCH_DIR}/include
		${BASE_SEARCH_DIR}/include/opencv4
		${PYTHON_INCLUDE_PATH}
	)

	link_directories(
		${BASE_SEARCH_DIR}/lib
	)

	add_library(pyautograff_utils SHARED
		${src}
	)

	#${PROJECT_SOURCE_DIR}/../../../../colormotor/addons/pycolormotor/modules
	target_link_libraries(pyautograff_utils
		#${COLORMOTOR_PATH}/addons/pycolormotor/modules/_cm.so
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
	add_library(pyautograff_utils SHARED
		${src}
	)

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










