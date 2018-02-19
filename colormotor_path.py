import os, sys
try:
    cmpath = os.environ['COLORMOTOR_PATH']
except KeyError:
    print('Could not find environment variable COLORMOTOR_PATH, set it in bash profile file')
    print('with the desiderd path for colormotor root dir')
modpath = os.path.abspath(os.path.join(cmpath, 'addons/pycolormotor/modules'))
if not modpath in sys.path:
    print('Adding colormotor path: ' + modpath)
    sys.path.append(modpath)

