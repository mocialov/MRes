# -*- encoding: UTF-8 -*- 

'''Cartesian control: Arm trajectory example'''

import sys
import motion
from naoqi import ALProxy
import time
import almath
import random


def SetStiffness(proxy, flag):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = flag
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def main(robotIP):
    ''' Example showing a hand ellipsoid
    Warning: Needs a PoseInit before executing
    '''

    try:
        motionProxy = ALProxy("ALMotion", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALMotion"
        print "Error was: ", e

    try:
        postureProxy = ALProxy("ALRobotPosture", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALRobotPosture"
        print "Error was: ", e

    # Set NAO in Stiffness On
    SetStiffness(motionProxy, 1)

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.9)
    
    effector  = "LArm"
    space  = 0 # TORSO_SPACE
    useSensorValues  = False
    result = motionProxy.getTransform(effector, space, useSensorValues)
    #R R R x
    #R R R y
    #R R R z
    #0 0 0 1
    #print "Transform of RArm"
    #print result
    #initial: 0.9466429352760315, -0.2590537667274475, 0.1917247474193573, 0.14743731915950775, -0.06496888399124146, 0.42928797006607056, 0.9008278846740723, 0.09203990548849106, -0.31566792726516724, -0.8652185201644897, 0.3895519971847534, -0.015188712626695633, 0.0, 0.0, 0.0, 1.0]
    
    space      = motion.FRAME_ROBOT
    isAbsolute = False

    effectorList = ["LArm", "RArm"]

    # Motion of Arms with block process
    axisMaskList = [7, 7]

    timesList    = [[5.0], [5.0]]         # seconds
    pathList     = [#front,sideways,up
     
     [[-0.04+random.uniform(-1, 1)/50.0, 0.17+random.uniform(-1, 1)/50.0, 0.1+random.uniform(-1, 1)/50.0, 0.0, 0.0, 0.0]],
     
     [[-0.04+random.uniform(-1, 1)/50.0, -0.17+random.uniform(-1, 1)/50.0, 0.1+random.uniform(-1, 1)/50.0, 0.0, 0.0, 0.0]]
                    ]
    motionProxy.positionInterpolations(effectorList, space, pathList,
                                       axisMaskList, timesList, isAbsolute)

    #shine eyes
	# Replace "127.0.0.1" with the IP of your NAO
    leds = ALProxy("ALLeds",robotIp,9559)
	# Turn the green face LEDs half on
    leds.fadeRGB("FaceLeds", 0x0000FF00, 1.0)

    time.sleep(10)

    leds.fadeRGB("FaceLeds", 0xFF000000, 1.0)

    #bring back
    postureProxy.goToPosture("StandInit", 0.9)

    #remove stiffness
    SetStiffness(motionProxy, 0)


if __name__ == "__main__":
    robotIp = "137.195.108.173"

    if len(sys.argv) <= 1:
        print "Usage python motion_cartesianArm2.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
