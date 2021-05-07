rm autograff_utils_wrap.h
rm autograff_utils_wrap.cxx
rm autograff_utils.py

echo "swig:wrapping autograff_utils..."

swig -w322,362 -python -c++ -extranative  -I./ -I./../../../../colormotor/addons/pycolormotor autograff_utils.i

echo "void initializeSwig_autograff_utils() { SWIG_init(); }" >> autograff_utils_wrap.cxx
echo "void initializeSwig_autograff_utils();" >> autograff_utils_wrap.h

echo "copying files"
# cp app.py ../../modules/app.py
cp autograff_utils.py ../autograff_utils.py

