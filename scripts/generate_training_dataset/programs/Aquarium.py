import numpy as np
import scipy
from scipy.io import loadmat
import cv2 as cv
import math
import copy
from skimage.util import random_noise
from programs.programsForDrawingImage import *
from programs.programsForGeneratingRandomFishes import *

class Bounding_Box:
    """
        Class that represents the bounding box of the fish

        Static Properties:
            areaThreshold: value representing the minimal area that a bounding box can have in order to be considered
                            a fish that can hold annotations, used for ignoring barely visible fish in the edges aswell
                            as adding patchy noise
        Properties:
            smallX (int): smallest x value of the bounding box
            smallY (int): smallest y value of the bounding box
            bigX (int): biggest x value of the bounding box
            bigY (int): biggest y value of the bounding box

        Methods:
            __init__, args(int, int, int, int): creates an object representing the bounding box of the fish,
                                                given the smallest x value of the box, the smallest y value of the box
                                                the biggest x value of the box, and the biggest Y value of the box

            CenterX: property representing the x coordinate of the center of teh bounding box
            CenterY: property representing y coordinate of the center of the bounding box
            Width: property representing the width of the bounding box
            Height: property representing height of the bounding box
            Area: property representing the area of the bounding box
            isBigEnough(): returns the True or False depending on whether the area of the bounding box is
                            greater that the threshold
    """
    areaThreshold = 25

    def __init__(self, smallX = 0, smallY = 0, bigX = 0, bigY = 0):
        self.smallX = smallX
        self.smallY = smallY
        self.bigX = bigX
        self.bigY = bigY
    
    @property
    def CenterX(self):
        return int(roundHalfUp((self.bigX + self.smallX) / 2))
    
    @property
    def CenterY(self):
        return int(roundHalfUp((self.bigY + self.smallY) / 2))
    
    @property
    def Width(self):
        width = (self.bigX - self.smallX)
        # Not really considered a box if it is out of bounds
        return (width if width > 0 else 0)

    @property
    def Height(self):
        height = (self.bigY - self.smallY)
        # Not really considered a box if it is out of bounds
        return (height if height > 0 else 0)
    
    @property
    def Area(self):
                return self.getWidth() * self.getHeight()
    
    def isBigEnough(self):
        return True if self.getArea() > Aquarium.Fish.Bounding_Box.areaThreshold else False



class Point:
    """
        Class used to represent the keypoints

        Properties:
            x (float): x coordinate
            y (float): y coordinate
            z (float): z/depth coordinate
            visibility (boolean): state representing whether or not the point is in the image

        Methods:
            __init__, args(float, float, float): creates a point with coordinates (x,y, z/depth)
                                                and sets visibility to True of False depending on whether that
                                                point is visible in the image
            XInt: property representing the x coordinate rounded to an integer
            YInt: property representing the y coordinate rounded to an integer
            isInBounds args(): returns true of false if the integer version of the point is in the image
    """

    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z
        # not calling isInBounds, because object may need to initialize first
        self.visibility = True if (x < imageSizeX and x >= 0 and y < imageSizeY and y >= 0) else False
    
    @property
    def XInt(self):
        return int(roundHalfUp(self.x))
    
    @property
    def YInt(self):
        return int(roundHalfUp(self.y))

    def isInBounds(self):
        xInt = self.getXInt()
        yInt = self.getYInt()
        return True if (xInt < imageSizeX and xInt >= 0 and yInt < imageSizeY and yInt >= 0) else False


class Fish:
    """
        Class representing a fish

        Static Properties:
            number_of_keypoints(int):  number of keypoints, set to 12
            number_of_backbone_points: number of backbone_points, set to 10

            proj_params (string): path to mat file containing projection parameters
            lut_b_tail = path to mat file containing the look up table for bottom view
            lut_s_tail = path to mat file containing the look up table for side view
            fish_shapes = path to mat file containing the look up table for fish positions
            lut_b_tail_mat (dict): dictionary of loaded lut_b
            lut_b_tail (numpy array): numpy array of lut_b
            lut_s_tail_mat (dict): dictionary of loaded lut_s
            lut_s_tail numpy array): numpy array of lut_s

        Properties:
            x (numpy array): 22 parameter vector representing the fish position
            fishlen (float): fish length

            graysContainer (list): containing the grayscale images from the three camera views: 'B', 'S1', 'S2'
            depthsContainer (list): containing depth for each image
            annotationsType (AnnotationsType): type depicting which annotations are desired

            # For keypoint annotations, default
            keypointsListContainer (list): Container for the keypoints as seen from each camera view
            boundingBoxContainer (list): Container for the bounding boxes as seen from each camera view

            # For segmentation annotations
            contourContainer (list): Container for the contours of the fish as seen from each camera view

            # TODO: check if these should be deleted
            cropsContainer (list): Container for the crops of the images from each camera view
            coor_3d (numpy array): coordinates in 3d space

        Methods:
            __init__, args(float, numpy array, AnnotationsType = keypoint): creates an instance of a fish as depicted by
                its 22 parameter vector an length.  It also sets the fish instance with the annotations you want, by
                default it is set to keypoint annotations.

            draw(): generates the images of the fish as seen from all three camera views.  It assigns values to the
                graysContainer,  depthsContainer, keypointsListContainer/contourContainer, and boundingBoxContainer.
    """
    # Static properties of Fish Class
    number_of_keypoints = 12
    number_of_backbone_points = 10

    def __init__(self, fishlen, x, annotationsType = AnnotationsType.keypoint):
        self.x = x
        self.fishlen = fishlen

        self.graysContainer = []
        self.depthsContainer = []
        self.annotationsType = annotationsType

        # For keypoint annotations, default
        self.keypointsListContainer = [[], [], []]
        self.boundingBoxContainer = [Fish.Bounding_Box(),
                                     Fish.Bounding_Box(),
                                     Fish.Bounding_Box()]

        # For segmentation annotations
        self.contourContainer = [None, None, None]

        # Old version of the script used these
        self.cropsContainer = []
        self.coor_3d = None

    def draw(self):
        [gray_b, gray_s1, gray_s2, crop_b, crop_s1, crop_s2, c_b, c_s1, c_s2, eye_b, eye_s1, eye_s2, self.coor_3d] = \
            return_graymodels_fish(self.x, Aquarium.Fish.lut_b_tail, Aquarium.Fish.lut_s_tail, Aquarium.Fish.proj_params, self.fishlen,
                                   imageSizeX, imageSizeY)

        self.graysContainer = [gray_b, gray_s1, gray_s2]
        self.cropsContainer = [crop_b, crop_s1, crop_s2]

        if self.annotationsType == AnnotationsType.segmentation:
            for viewIdx in range(3):
                ########################
                gray = self.graysContainer[viewIdx]
                gray = gray.astype(np.uint8)
                contours = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                contours = contours[0]
                numberOfContours = len(contours)
                if numberOfContours != 0:
                    contourToAdd = None
                    maxAmountOfPoints = 0

                    for contourIdx in range(numberOfContours):
                        amountOfPoints = contours[contourIdx].shape[0]
                        if maxAmountOfPoints <= amountOfPoints:
                            maxAmountOfPoints = amountOfPoints
                            contourToAdd = contours[contourIdx][:, 0, :]

                    if contourToAdd is not None:
                        self.contourContainer[viewIdx] = contourToAdd

        coorsContainer = [c_b, c_s1, c_s2]
        eyesContainer = [eye_b, eye_s1, eye_s2]
        depthsContainer = [imageSizeY - c_s1[1, :], c_b[0, :], c_b[1, :]]
        for viewIdx in range(3):
            coors = coorsContainer[viewIdx]
            eyes = eyesContainer[viewIdx]
            pointsInViewList = self.keypointsListContainer[viewIdx]
            depths = depthsContainer[viewIdx]
            for pointIdx in range(Aquarium.Fish.number_of_backbone_points):
                x = coors[0, pointIdx]
                y = coors[1, pointIdx]
                z = depths[pointIdx]
                point = Aquarium.Fish.Point(x, y, z)
                pointsInViewList.append(point)
            for pointIdx in range(2):
                x = eyes[0, pointIdx]
                y = eyes[1, pointIdx]
                point = Aquarium.Fish.Point(x, y)
                pointsInViewList.append(point)

            # Updating, sometimes necessary, will have to check if it is in this case
            self.keypointsListContainer[viewIdx] = pointsInViewList
        # self.keypointsListContainer = [pointsInViewBList, keypointsListContainer[0], keypointsListContainer[1]]


class Aquarium:
    def __init__():
        proj_params = 'proj_params_101019_corrected_new'
        proj_params = inputsFolder + proj_params

        lut_b_tail = 'lut_b_tail.mat'
        lut_s_tail = 'lut_s_tail.mat'
        fish_shapes = 'generated_pose_100_percent.mat'
        fish_shapes = inputsFolder + fish_shapes

        lut_s_tail_mat = loadmat(inputsFolder + lut_b_tail)
        lut_b_tail = lut_s_tail_mat['lut_b_tail']

        lut_s_tail_mat = loadmat(inputsFolder + lut_s_tail)
        lut_s_tail = lut_s_tail_mat['lut_s_tail']
