from controller import Supervisor, Accelerometer, Camera, DistanceSensor, \
                       GPS, Gyro, InertialUnit, Keyboard, LED, Motion, \
                       Motor, TouchSensor, Node, Field
import numpy as np

# this is the main class
class Nao(Supervisor):
    PHALANX_MAX = 8

    # the accelerometer axes are oriented as on the real robot
    # however the sign of the returned values may be opposite
    def getAcceleration(self):
        acc = [0, 0, 0]
        acc_real = self.accelerometer.getValues()
        min = -20
        max = 20
        acc[0] = ((acc_real[0] - min)/(max-min)) * 2 - 1
        acc[1] = ((acc_real[1] - min)/(max-min)) * 2 - 1
        acc[2] = ((acc_real[2] - min)/(max-min)) * 2 - 1
        return acc[0], acc[1], acc[2]

    # the gyro axes are oriented as on the real robot
    # however the sign of the returned values may be opposite
    def getGyroscope(self):
        vel = [0, 0]
        vel_real = self.gyro.getValues()
        min = -6
        max = 6
        vel[0] = ((vel_real[0] - min)/(max-min)) * 2 - 1
        vel[1] = ((vel_real[1] - min)/(max-min)) * 2 - 1
        return vel[0], vel[1]

    def getPos(self):
        p = self.gps.getValues()
        #print('----------gps----------')
        #print('position: [ x y z ] = [%f %f %f]' % (p[0], p[1], p[2]))
        return p[0], p[1], p[2]

    # the InertialUnit roll/pitch angles are equal to naoqi's AngleX/AngleY
    def getRPY(self):
        rpy = [0, 0, 0]
        rpy_real = self.inertialUnit.getRollPitchYaw()
        min = -3.14
        max = 3.14
        rpy[0] = ((rpy_real[0] - min)/(max-min)) * 2 - 1
        rpy[1] = ((rpy_real[1] - min)/(max-min)) * 2 - 1
        return rpy[0], rpy[1], rpy[2]

    def getFootSensors(self):
        newtons = 0.0
        fsv = [] # force sensor values
        fsv.append(self.fsr[0].getValues())
        fsv.append(self.fsr[1].getValues())
        l = []
        r = []
        #newtonsLeft = 0
        #newtonsRight = 0
        # The coefficients were calibrated against the real
        # robot so as to obtain realistic sensor values.
        l.append(fsv[0][2] / 3.4 + 1.5 * fsv[0][0] + 1.15 * fsv[0][1]) # Left Foot Front Left
        l.append(fsv[0][2] / 3.4 + 1.5 * fsv[0][0] - 1.15 * fsv[0][1]) # Left Foot Front Right
        l.append(fsv[0][2] / 3.4 - 1.5 * fsv[0][0] - 1.15 * fsv[0][1]) # Left Foot Rear Right
        l.append(fsv[0][2] / 3.4 - 1.5 * fsv[0][0] + 1.15 * fsv[0][1]) # Left Foot Rear Left
        r.append(fsv[1][2] / 3.4 + 1.5 * fsv[1][0] + 1.15 * fsv[1][1]) # Right Foot Front Left
        r.append(fsv[1][2] / 3.4 + 1.5 * fsv[1][0] - 1.15 * fsv[1][1]) # Right Foot Front Right
        r.append(fsv[1][2] / 3.4 - 1.5 * fsv[1][0] - 1.15 * fsv[1][1]) # Right Foot Rear Right
        r.append(fsv[1][2] / 3.4 - 1.5 * fsv[1][0] + 1.15 * fsv[1][1]) # Right Foot Rear Left
        for i in range(0, len(l)):
            mi = 0
            ma = 25
            l[i] = max(min(l[i], ma), mi)
            #newtonsLeft += l[i]
            l[i] = ((l[i] - mi)/(ma-mi))
            r[i] = max(min(r[i], ma), mi)
            #newtonsRight += r[i]
            r[i] = ((r[i] - mi)/(ma-mi))

        #print('total: %f Newtons, %f kilograms' % ((newtonsLeft + newtonsRight), ((newtonsLeft + newtonsRight)/9.81)))
        return l, r

    def getFootBumpers(self):
        ll = self.lfootlbumper.getValue()
        lr = self.lfootrbumper.getValue()
        rl = self.rfootlbumper.getValue()
        rr = self.rfootrbumper.getValue()

        return ll, lr, rl, rr

    def getUltrasoundSensors(self):
        dist = []
        min = 0
        max = 2.55
        for i in range(0, len(self.us)):
            d_real = self.us[i].getValue()
            d = ((d_real - min)/(max-min))
            dist.append(d)
        return dist[0], dist[1]


    def printCameraImage(self, camera):
        scaled = 2 # defines by which factor the image is subsampled
        width = camera.getWidth()
        height = camera.getHeight()

        # read rgb pixel values from the camera
        image = camera.getImage()

        for y in range(0, height/scaled):
            line = ''
            for x in range(0, width/scaled):
                gray = camera.imageGetGray(image, width, x * scaled, y * scaled) * 9 / 255 # between 0 and  instead of 0 and 255
                line = line + str(int(gray))
            print(line)

    def setAllLedsColor(self, rgb):
        # these leds take RGB values
        for i in range(0, len(self.leds)):
            self.leds[i].set(rgb)

        # ear leds are single color (blue)
        # and take values between 0 - 255
        self.leds[5].set(rgb & 0xFF)
        self.leds[6].set(rgb & 0xFF)

    def setJointPositions(self, positions):
        #expects positions in range [-1,1] sets positions in MotorLimits
        for motor, pos in positions.items():
            min,max = self.motorLimits[motor]
            if min > 0:
                min = min - 0.00001
            if max > 0:
                max = max - 0.00001
            real_pos = ((pos + 1) / 2) * (max - min) + min
            self.getMotor(motor).setPosition(real_pos)

    def setHandsAngle(self, angle):
        for i in range(0, self.PHALANX_MAX):
            clampedAngle = angle
            if clampedAngle > self.maxPhalanxMotorPosition[i]:
                clampedAngle = self.maxPhalanxMotorPosition[i]
            elif clampedAngle < self.minPhalanxMotorPosition[i]:
                clampedAngle = self.minPhalanxMotorPosition[i]

            if len(self.rphalanx) > i and self.rphalanx[i] is not None:
                self.rphalanx[i].setPosition(clampedAngle)
            if len(self.lphalanx) > i and self.lphalanx[i] is not None:
                self.lphalanx[i].setPosition(clampedAngle)

    def findAndEnableDevices(self):
        # enable battery
        # self.batterySensorEnable(self, self.timeStep) does not work by now

        # get the time step of the current world.
        self.timeStep = int(self.getBasicTimeStep())
        print('World´s timestep', self.timeStep)

        # camera
        self.cameraTop = self.getCamera("CameraTop")
        self.cameraBottom = self.getCamera("CameraBottom")
        #self.cameraTop.enable(self.timeStep)
        #self.cameraBottom.enable(self.timeStep)

        # accelerometer
        self.accelerometer = self.getAccelerometer('accelerometer')
        self.accelerometer.enable(self.timeStep)

        # gyro
        self.gyro = self.getGyro('gyro')
        self.gyro.enable(self.timeStep)

        # gps
        self.gps = self.getGPS('gps')
        self.gps.enable(self.timeStep)

        # inertial unit
        self.inertialUnit = self.getInertialUnit('inertial unit')
        self.inertialUnit.enable(self.timeStep)

        # ultrasound sensors
        self.us = []
        usNames = ['Sonar/Left','Sonar/Right']
        for i in range(0, len(usNames)):
            self.us.append(self.getDistanceSensor(usNames[i]))
            self.us[i].enable(self.timeStep)

        # foot sensors
        self.fsr = []
        fsrNames = ['LFsr', 'RFsr']
        for i in range(0, len(fsrNames)):
            self.fsr.append(self.getTouchSensor(fsrNames[i]))
            self.fsr[i].enable(self.timeStep)

        # foot bumpers
        self.lfootlbumper = self.getTouchSensor('LFoot/Bumper/Left')
        self.lfootrbumper = self.getTouchSensor('LFoot/Bumper/Right')
        self.rfootlbumper = self.getTouchSensor('RFoot/Bumper/Left')
        self.rfootrbumper = self.getTouchSensor('RFoot/Bumper/Right')
        self.lfootlbumper.enable(self.timeStep)
        self.lfootrbumper.enable(self.timeStep)
        self.rfootlbumper.enable(self.timeStep)
        self.rfootrbumper.enable(self.timeStep)

        # there are 7 controlable LED groups in Webots
        self.leds = []
        self.leds.append(self.getLED('ChestBoard/Led'))
        self.leds.append(self.getLED('RFoot/Led'))
        self.leds.append(self.getLED('LFoot/Led'))
        self.leds.append(self.getLED('Face/Led/Right'))
        self.leds.append(self.getLED('Face/Led/Left'))
        self.leds.append(self.getLED('Ears/Led/Right'))
        self.leds.append(self.getLED('Ears/Led/Left'))

        # get phalanx motor tags
        # the real Nao has only 2 motors for RHand/LHand
        # but in Webots we must implement RHand/LHand with 2x8 motors
        self.lphalanx = []
        self.rphalanx = []
        self.maxPhalanxMotorPosition = []
        self.minPhalanxMotorPosition = []
        for i in range(0, self.PHALANX_MAX):
            self.lphalanx.append(self.getMotor("LPhalanx%d" % (i + 1)))
            self.rphalanx.append(self.getMotor("RPhalanx%d" % (i + 1)))

            # assume right and left hands have the same motor position bounds
            self.maxPhalanxMotorPosition.append(self.rphalanx[i].getMaxPosition())
            self.minPhalanxMotorPosition.append(self.rphalanx[i].getMinPosition())

    def getMotorsLimits(self):
        #returns a dict with the name of the motor as key and a tuple with Limits (min, max)
        deviceList = []
        jointLimits = dict()

        for i in range(self.devicesNumber):
            deviceType = self.getDeviceByIndex(i).getNodeType()
            if deviceType == 52:
                name = self.getDeviceByIndex(i).getName()
                self.motor_device[name] = self.getDeviceByIndex(i)
                min = self.motor_device[name].getMinPosition()
                max = self.motor_device[name].getMaxPosition()
                jointLimits[name] = (min,max)

                #enable torque feedback
                self.motor_device[name].enableTorqueFeedback(self.timeStep)
                #print('Torque feedback enabled ',self.motor_device[name].getForceFeedbackSamplingPeriod())

        return jointLimits

    def getMotorSensorsNames(self):
        motorSensorsNames = []
        for i in range(self.devicesNumber):
            deviceType = self.getDeviceByIndex(i).getNodeType()
            if deviceType == 47:
                name = self.getDeviceByIndex(i).getName()
                self.getPositionSensor(name).enable(self.timeStep)
                motorSensorsNames.append(name)
        return motorSensorsNames

    def readMotorPosition(self):
        readings = list()
        for s in self.motorSensorsNames:
            m = s[0:-1] #delete  the last "S" in the sensor name to obtain motor name
            real_pos = self.getPositionSensor(s).getValue()
            min = self.motorLimits[m][0]
            max = self.motorLimits[m][1]
            self.motor_device[m].getTorqueFeedback()
            pos = ((real_pos - min) / (max-min) ) * 2 - 1
            readings.append(pos)
        return readings

    def getAllReadings(self):

        readings = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        readings[0:2] = self.getAcceleration()
        readings[3:4] = self.getGyroscope()
        rpy1,rpy2,rpy3= self.getRPY()
        readings[5] = rpy1
        readings[6] = rpy2
        l,r = self.getFootSensors()
        readings[7:10] = l
        readings[11:14] = r
        readings[15:18] = self.getFootBumpers()
        readings[19:20] = self.getUltrasoundSensors()
        motors = self.readMotorPosition()
        #print(motors)
        for m in motors:
            readings.append(m)
        return readings

    def getTorques(self):
        torque = dict()
        for name in self.motor_names:
            torque[name] = self.motor_device[name].getTorqueFeedback()
        return torque

    def resetRobotPosition(self):
        #tempZip = zip(self.motor_names, self.INITIAL_MOTOR_POS)
        #jointPositions = dict(tempZip)
        #random_init_pos = dict()
        #for k in self.DEFAULT_MOTOR_POS.keys():
        #    self.INITIAL_MOTOR_POS[k] = max(min(self.DEFAULT_MOTOR_POS[k] + np.random.uniform(low=-0.05, high = 0.05), 1), -1)
            #print(k,self.INITIAL_MOTOR_POS[k])
        self.setJointPositions(self.INITIAL_MOTOR_POS)
        for i in range(100):
            self.step(self.timeStep)
        Field.setSFVec3f(self.trans_field, self.INITIAL_TRANS)
        Field.setSFRotation(self.rot_field, self.INITIAL_ROT)
        #print(jointPositions)
        #wb_supervisor_simulation_reset_physics
        Supervisor.simulationResetPhysics(self)
        #print('-------------------------------------')
        #print('>>>>>>Robot pos has been reset<<<<<<')
        #print('-------------------------------------')
        return True

    def getRobotVelocity(self):
        return self.sup.getVelocity()

    def getRobotPosition(self):
        return self.sup.getPosition()

    def __init__(self):
        Supervisor.__init__(self)
        self.currentlyPlaying = False
        self.devicesNumber = self.getNumberOfDevices()
        self.sup = self.getSelf()
        # initialize stuff
        self.findAndEnableDevices()
        # get motors and its limits:
        self.motor_device = dict()
        self.motorLimits = self.getMotorsLimits()
        self.motorSensorsNames = self.getMotorSensorsNames()
        self.robot_node = Supervisor.getSelf(self)
        self.trans_field = Node.getField(self.robot_node,'translation')
        self.rot_field = Node.getField(self.robot_node,'rotation')
        # self.INITIAL_TRANS = Field.getSFVec3f(self.trans_field)
        # self.INITIAL_ROT = Field.getSFRotation(self.rot_field)
        self.INITIAL_TRANS = [0, 0.345, 0]
        self.INITIAL_ROT = [-1, 0, 0, 1.5708]
        self.motor_names = list(self.motorLimits.keys())
        self.INITIAL_MOTOR_POS = {'HeadYaw': 0.0,
                                'HeadPitch': 0.13235322780693037,
                                'RShoulderPitch': 0.8,
                                'RShoulderRoll': 0.75,
                                'RElbowYaw': 0.8,
                                'RElbowRoll': -0.1,
                                'RWristYaw': -3.43941389813196e-08,
                                'RPhalanx1': -1.0,
                                'RPhalanx2': -1.0,
                                'RPhalanx3': -1.0,
                                'RPhalanx4': -1.0,
                                'RPhalanx5': -1.0,
                                'RPhalanx6': -1.0,
                                'RPhalanx7': -1.0,
                                'RPhalanx8': -1.0,
                                'LShoulderPitch': 0.8,
                                'LShoulderRoll': -0.75,
                                'LElbowYaw': -0.8,
                                'LElbowRoll': 0.1,
                                'LWristYaw': 0.0,
                                'LPhalanx1': -1.0,
                                'LPhalanx2': -1.0,
                                'LPhalanx3': -1.0,
                                'LPhalanx4': -1.0,
                                'LPhalanx5': -1.0,
                                'LPhalanx6': -1.0,
                                'LPhalanx7': -1.0,
                                'LPhalanx8': -1.0,
                                'RHipYawPitch': 0.25,
                                'RHipRoll': 0.25,
                                'RHipPitch': 0.5,
                                'RKneePitch': -0.8,
                                'RAnklePitch': -0.52,
                                'RAnkleRoll': 0.35,
                                'LHipYawPitch': 0.25,
                                'LHipRoll': -0.25,
                                'LHipPitch': 0.25,
                                'LKneePitch': -0.8,
                                'LAnklePitch': 0.37,
                                'LAnkleRoll': -0.35}
        self.LEFT_STEP = {'HeadYaw': 0.0,
                                'HeadPitch': 0.13235322780693037,
                                'RShoulderPitch': 0.8,
                                'RShoulderRoll': 0.75,
                                'RElbowYaw': 0.8,
                                'RElbowRoll': -0.1,
                                'RWristYaw': -3.43941389813196e-08,
                                'RPhalanx1': -1.0,
                                'RPhalanx2': -1.0,
                                'RPhalanx3': -1.0,
                                'RPhalanx4': -1.0,
                                'RPhalanx5': -1.0,
                                'RPhalanx6': -1.0,
                                'RPhalanx7': -1.0,
                                'RPhalanx8': -1.0,
                                'LShoulderPitch': 0.8,
                                'LShoulderRoll': -0.75,
                                'LElbowYaw': -0.8,
                                'LElbowRoll': 0.1,
                                'LWristYaw': 0.0,
                                'LPhalanx1': -1.0,
                                'LPhalanx2': -1.0,
                                'LPhalanx3': -1.0,
                                'LPhalanx4': -1.0,
                                'LPhalanx5': -1.0,
                                'LPhalanx6': -1.0,
                                'LPhalanx7': -1.0,
                                'LPhalanx8': -1.0,
                                'RHipYawPitch': 0.25,
                                'RHipRoll': 0.25,
                                'RHipPitch': 0.85,
                                'RKneePitch': -0.8,
                                'RAnklePitch': -0.37,
                                'RAnkleRoll': 0.35,
                                'LHipYawPitch': 0.25,
                                'LHipRoll': -0.25,
                                'LHipPitch': 0,
                                'LKneePitch': -0.8,
                                'LAnklePitch': 0.52,
                                'LAnkleRoll': -0.35}
        self.INITIAL_COPY = {'HeadYaw': 0.0,
                                'HeadPitch': 0.13235322780693037,
                                'RShoulderPitch': 0.8,
                                'RShoulderRoll': 0.75,
                                'RElbowYaw': 0.8,
                                'RElbowRoll': -0.1,
                                'RWristYaw': -3.43941389813196e-08,
                                'RPhalanx1': -1.0,
                                'RPhalanx2': -1.0,
                                'RPhalanx3': -1.0,
                                'RPhalanx4': -1.0,
                                'RPhalanx5': -1.0,
                                'RPhalanx6': -1.0,
                                'RPhalanx7': -1.0,
                                'RPhalanx8': -1.0,
                                'LShoulderPitch': 0.8,
                                'LShoulderRoll': -0.75,
                                'LElbowYaw': -0.8,
                                'LElbowRoll': 0.1,
                                'LWristYaw': 0.0,
                                'LPhalanx1': -1.0,
                                'LPhalanx2': -1.0,
                                'LPhalanx3': -1.0,
                                'LPhalanx4': -1.0,
                                'LPhalanx5': -1.0,
                                'LPhalanx6': -1.0,
                                'LPhalanx7': -1.0,
                                'LPhalanx8': -1.0,
                                'RHipYawPitch': 0.25,
                                'RHipRoll': 0.25,
                                'RHipPitch': 0.5,
                                'RKneePitch': -0.8,
                                'RAnklePitch': 0.1,
                                'RAnkleRoll': 0.4,
                                'LHipYawPitch': 0.25,
                                'LHipRoll': -0.25,
                                'LHipPitch': 0.5,
                                'LKneePitch': -0.8,
                                'LAnklePitch': 0.1,
                                'LAnkleRoll': -0.4}
        self.DEFAULT_MOTOR_POS = self.INITIAL_MOTOR_POS
