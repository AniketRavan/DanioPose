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
            return_graymodels_fish(self.x, Aquarium.lut_b_tail, Aquarium.lut_s_tail, Aquarium.proj_params, self.fishlen, imageSizeX, imageSizeY)

        self.graysContainer = [gray_b, gray_s1, gray_s2]
        self.cropsContainer = [crop_b, crop_s1, crop_s2]

        if self.annotationsType == AnnotationsType.segmentation:
            for viewIdx in range(3):
                ########################
                gray = self.graysContainer[viewIdx]
                gray = gray.astype(np.uint8)
                contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
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
            for pointIdx in range(Fish.number_of_backbone_points):
                x = coors[0, pointIdx]
                y = coors[1, pointIdx]
                z = depths[pointIdx]
                point = Fish.Point(x, y, z)
                pointsInViewList.append(point)
            for pointIdx in range(2):
                x = eyes[0, pointIdx]
                y = eyes[1, pointIdx]
                point = Fish.Point(x, y)
                pointsInViewList.append(point)

            # Updating, sometimes necessary, will have to check if it is in this case
            self.keypointsListContainer[viewIdx] = pointsInViewList
        # self.keypointsListContainer = [pointsInViewBList, keypointsListContainer[0], keypointsListContainer[1]]

        # Creating Depth Arrays, Updating Eyes, and getting their bounding boxes
        for viewIdx in range(3):
            gray = self.graysContainer[viewIdx]
            keypointsList = self.keypointsListContainer[viewIdx]
            # TODO modify this later so that it also includes the eyes
            xArr = np.array([point.x for point in keypointsList[:Aquarium.Fish.number_of_backbone_points]])
            yArr = np.array([point.y for point in keypointsList[:Aquarium.Fish.number_of_backbone_points]])
            zArr = np.array([point.z for point in keypointsList[:Aquarium.Fish.number_of_backbone_points]])
            #  Creating Depth Arrays  img,  y coor,   x coor,   depth coor
            depthIm = createDepthArr(gray, yArr, xArr, zArr)
            (self.depthsContainer).append(depthIm)

            # TODO fix this so that I dont have to update it like this
            for pointIdx, point in enumerate(keypointsList):
                if point.isInBounds():
                    point.z = depthIm[point.getYInt(), point.getXInt()]
                    point.visibility = True
                else:
                    point.visibility = False
                keypointsList[pointIdx] = point

            # Updating, # TODO check if this is necessary
                self.keypointsListContainer[viewIdx] = keypointsList

            # Finding the bounding box
            nonZeroPoints = np.argwhere(gray != 0)
            if len(nonZeroPoints) != 0:
                [smallY, smallX, bigY, bigX] = [np.min(nonZeroPoints[:, 0]), np.min(nonZeroPoints[:, 1]),
                                                np.max(nonZeroPoints[:, 0]), np.max(nonZeroPoints[:, 1])]

                boundingBox = Aquarium.Fish.Bounding_Box(smallX, smallY, bigX, bigY)
                self.boundingBoxContainer[viewIdx] = boundingBox


class Aquarium:
    """
        Class representing a container for the fishes.  It serves to merge all the images of the fishes aswell as get
        the annotations of the fishes.

        Static Properties:
            maxAmountOfFishInAllViews (int): the maximum amount of fishes that can be visible in all views.
            patchy_noise (boolean): True or False representing whether or not to add patchy noise
            amountOfData (int): Amount of data that will be generated, used to split the training data set
            fractionToBe4Training (float): Fraction representing the amount of data that should be used for training
            biggestIdx4TrainingData (float): amountOfData * fractionToBe4Training, the biggest Idx that will be used
                to for Training data

            imagesSubFolder (string): sub path to the images folder
            labelsSubFolder (string): sub path to the labels folder

        Properties:
            idx (int): number representing the index of which data file is being generated
                    self.dataFolderPath = dataFolderPath
            dataFolderPath (string): path to the parent folder in which the data will be saved to
            fishList (list): list that will contain fish objects
            allGContainer (list): list that will contain all the grayscale images
            finalGraysContainer (list): container with the merged images, len of 3 for the three camera views
            allGDepthContainer (list): list that will contain the depth of the grayscale images
            finalDepthsContainer (list): container with the merged depth, len of 3 for the three camera views,
                usefull for updating the visibility of the fish keypoints
            amountOfFish (int): variable for the amount of fishes
            reflectedFishContainer (list): of size three for consistency, but only the 2nd and 3rd element are really
                used.  Need to keep track of the annotations.
            annotationsType (AnnotationsType): flag used to tell which type of annotations are needed

            reflectedFishOutlinesContainer (list): of size three for consistency, but only the 2nd and 3rd element are
                really used.  It contains the contours of the reflected fish

        Methods:
            __init__, args(int, AnnotationsType, dataFolderPath): Creates an instance of the aquarium class.  It
                initializes a lot of the variables and calls generateRandomConfiguration

            generateRandomConfiguration() -> None: Creates a random set of fishes, more info on the actual method
            draw() -> None: Computes the images of each of the three views resulting from the set of fishes
            addReflections() -> None: Add instances of reflected fishes if they are past the water level
            updateFishVisibility() -> None: Goes through each of the keypoint list and updates if the points ended up
                visible or not
            addNoise() -> None: adds static noise to each of the final images of the fishes
            addPatchyNoise() -> None: randomly draws noisy disks on the images to simulate fog
            getGrays() -> list: returns list containing the final images of the fishes
            saveGrays() -> None: saves the three images of the fishes as a rgb png
            createSegmentationAnnotations() -> None: creates segmentation annotations according to the YOLO format
            createKeypointAnnotations() -> None:  creates keypoints annotations according to the YOLO format
            saveAnnotations() -> None: calls createKeypointAnnotations or createSegmentationAnnotations
                depending on the annotations type
    """

    # Aquarium Class Static Properties
    maxAmountOfFishInAllViews = 7
    patchy_noise = True
    amountOfData = 50000
    fractionToBe4Training = .9
    biggestIdx4TrainingData = amountOfData * fractionToBe4Training

    imagesSubFolder = 'images/'
    labelsSubFolder = 'labels/'

    aquariumVariables = ['fishInAllViews','fishInB', 'fishInS1', 'fishInS2', 'fishInEdges','overlapping']
    fishVectListKey = 'fishVectList'

    def __init__(self):
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
        
        self.idx = idx
        self.dataFolderPath = dataFolderPath

        self.fishList = []
        self.allGContainer = []
        self.finalGraysContainer = []
        self.allGDepthContainer = []
        self.finalDepthsContainer = []
        self.amountOfFish = 0
        self.waterLevel = None
        self.reflectedFishContainer = [[], [], []]
        self.annotationsType = annotationsType

        # TODO: might have to be deleted
        self.reflectedFishOutlinesContainer = [[], [], []]

        # Detecting which type of aquarium you want to generate
        aquariumVariablesDict = {'fishInAllViews':0, 'fishInB':0, 'fishInS1':0, 'fishInS2':0, 'fishInEdges':0,
                                 'overlapping':0}
        wasAnAquariumVariableDetected = False
        wasAnAquariumPassed = False
        for key in kwargs:
            if key in Aquarium.aquariumVariables:
                aquariumVariablesDict[key] = kwargs.get(key)
                wasAnAquariumVariableDetected = True
            if key is Aquarium.fishVectListKey:
                wasAnAquariumPassed = True
                fishVectList = kwargs.get(key)
            if key == 'waterLever':
                self.waterLevel = kwargs.get(key)


        # Initializing the aquarium based on the args
        if not wasAnAquariumPassed:
            if wasAnAquariumVariableDetected:
                fishVectList = self.generateFishListGivenVariables(aquariumVariablesDict)
            else:
                fishVectList = self.generateRandomConfiguration()
        self.amountOfFish = len(fishVectList)


        # Create the list of fishes
        for fishVect in fishVectList:
            fishlen = fishVect[0]
            x = fishVect[1:]
            fish = Aquarium.Fish(fishlen, x, self.annotationsType)
            (self.fishList).append(fish)

        if self.waterLevel is None:
            # Set it to a random value
            self.waterLevel = 0
            shouldThereBeWater = True if np.random.rand() > .5 else False
            if shouldThereBeWater:
                self.waterLevel = np.random.randint(10, high=60)



    def generateFishListGivenVariables(self,aquariumVariablesDict):
        fishInAllViews = aquariumVariablesDict.get('fishInAllViews')
        fishInB = aquariumVariablesDict.get('fishInB')
        fishInS1 = aquariumVariablesDict.get('fishInS1')
        fishInS2 = aquariumVariablesDict.get('fishInS2')
        overlapping = aquariumVariablesDict.get('overlapping')
        fishInEdges = aquariumVariablesDict.get('fishInEdges')
        fishList = generateAquariumOverlapsAndWithoutBoundsAndEdges(fishInAllViews, fishInB, fishInS1,
                                                                    fishInS2, overlapping, fishInEdges )
        return fishList

    def generateRandomConfiguration(self):
        self.fishInAllView = np.random.randint(0, Aquarium.maxAmountOfFishInAllViews)

        self.overlapping = 0
        for jj in range(self.fishInAllView):
            shouldItOverlap = np.random.rand()
            if shouldItOverlap > .5:
                self.overlapping += 1

        self.fishesInB = np.random.poisson(2)
        self.fishesInS1 = np.random.poisson(2)
        self.fishesInS2 = np.random.poisson(2)

        self.fishesInEdges = 0
        if np.random.rand() > .5:
            self.fishesInEdges = np.random.poisson(3)


        self.amountOfFish = self.fishInAllView + self.overlapping + self.fishesInB \
                            + self.fishesInS1 + self.fishesInS2 + self.fishesInEdges

        self.s1FishReflected = []
        self.s2FishReflected = []

        aquarium = generateAquariumOverlapsAndWithoutBoundsAndEdges(self.fishInAllView, self.fishesInB, self.fishesInS1,
                                                                    self.fishesInS2, self.overlapping, self.fishesInEdges)

        return aquarium


    def draw(self):
        # TODO replacing these by adding another dimension to a numpy array might make it better
        for viewIdx in range(3):
            allG = np.zeros((self.amountOfFish, imageSizeY, imageSizeX ))
            allGDepth = np.zeros((self.amountOfFish, imageSizeY, imageSizeX))
            self.allGContainer.append(allG)
            self.allGDepthContainer.append( allGDepth)

        # Getting the grayscale images as well as depth images from the fishes
        for fishIdx, fish in enumerate(self.fishList):
            fish.draw()
            graysContainer = fish.graysContainer
            depthsContainer = fish.depthsContainer

            for viewIdx in range(3):
                gray = graysContainer[viewIdx]
                depth = depthsContainer[viewIdx]
                gContainer = self.allGContainer[viewIdx]
                depthContainer = self.allGDepthContainer[viewIdx]

                gContainer[fishIdx,...] = gray
                depthContainer[fishIdx, ...] = depth

                # Updating, TODO check if this is necessary
                self.allGContainer[viewIdx] = gContainer
                self.allGDepthContainer[viewIdx] = depthContainer

        if self.waterLevel > 0 :
            self.addReflections()

        # Merging all the images together
        for viewIdx in range(3):
            allG = self.allGContainer[viewIdx]
            allGDepth = self.allGDepthContainer[viewIdx]

            finalGray, finalDepth = mergeMultipleGrayswithEdgesBlurred(allG, allGDepth)
            self.finalGraysContainer.append(finalGray)
            self.finalDepthsContainer.append(finalDepth)
        # After merging all the pictures some parts of the fish may have become hidden
        self.updateFishVisibility()

        self.addNoise()
        if Aquarium.patchy_noise :
            self.addPatchyNoise()


    def addReflections(self):
        # Only viewS1 and viewS2
        for viewIdx in range(1,3):
            reflectedFishList = self.reflectedFishContainer[viewIdx]
            allG = self.allGContainer[viewIdx]
            allGDepth = self.allGDepthContainer[viewIdx]

            for fishIdx, fish in enumerate(self.fishList):
                boundingBox = fish.boundingBoxContainer[viewIdx]
                topPoint = int(boundingBox.smallY)

                if topPoint <= self.waterLevel and not(topPoint <= 0):

                    # The fish will have a reflection
                    # TODO update this so that you also consider when the fish falls within the range of the reflection
                    gray = fish.graysContainer[viewIdx]
                    depth = fish.depthsContainer[viewIdx]
                    keypointsList = fish.keypointsListContainer[viewIdx]
                    # Adding the reflections to the images
                    grayFlipped = np.flipud(np.copy(gray))
                    grayDepthFlipped = np.flipud(depth)


                    gray[0:topPoint, :] = np.copy(1 * grayFlipped[-topPoint * 2:-topPoint, :])
                    # graysContainer[viewIdx] = gray
                    depth[0:topPoint, :] = grayDepthFlipped[-topPoint * 2:-topPoint, :]
                    # graysDepthContainer[viewIdx] = depth
                    # Adding a reflected fish object for annotations
                    reflectedFish = copy.deepcopy(fish)
                    reflectedKeypoints = reflectedFish.keypointsListContainer[viewIdx]
                    for pointIdx , point in enumerate(reflectedKeypoints):
                        x = point.x
                        y = point.y
                        y = (imageSizeY - 1) - y
                        y -= imageSizeY - (2 * topPoint)
                        # These next three lines update
                        point.y = y
                        if point.isInBounds():
                            point.visibility = True
                            point.z = depth[point.getYInt(), point.getXInt()]
                        else:
                            point.visibility = False
                        reflectedKeypoints[pointIdx] = point
                    reflectedFish.keypointsListContainer[viewIdx] = reflectedKeypoints

                    reflectedBoundingBox = reflectedFish.boundingBoxContainer[viewIdx]
                    boundingBoxYs = np.array([reflectedBoundingBox.smallY, reflectedBoundingBox.bigY])
                    boundingBoxYs = (imageSizeY - 1) - boundingBoxYs
                    boundingBoxYs -= imageSizeY - (2 * topPoint)
                    [reflectedBoundingBox.bigY, reflectedBoundingBox.smallY] = boundingBoxYs
                    reflectedBoundingBox.smallY = np.clip(reflectedBoundingBox.smallY, 0, imageSizeY - 1)
                    reflectedBoundingBox.bigY = np.clip(reflectedBoundingBox.bigY, 0, imageSizeY - 1)
                    # Updating TODO check if this is really necessary
                    reflectedFish.boundingBoxContainer[viewIdx] = reflectedBoundingBox

                    # In case you want segmentation annotations
                    if self.annotationsType == AnnotationsType.segmentation:
                        tempMask = np.zeros((imageSizeY, imageSizeX))
                        tempMask[0:topPoint, :] = np.copy(1 * grayFlipped[-topPoint * 2:-topPoint, :])

                        tempMask = tempMask.astype(np.uint8)

                        contours = cv.findContours(tempMask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
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
                                reflectedFish.contourContainer[viewIdx] = contourToAdd

                    allG[fishIdx,...] = gray
                    allGDepth[fishIdx, ...] = depth
                    reflectedFishList.append(reflectedFish)

            # Updating TODO check if this is really necessary
            self.reflectedFishContainer[viewIdx] = reflectedFishList
            self.allGContainer[viewIdx] = allG
            self.allGDepthContainer[viewIdx] = allGDepth
