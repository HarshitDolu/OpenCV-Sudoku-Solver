import os
import numpy as np
import cv2
from utils import *
import pickle
import sudokuSolver
from matplotlib import pyplot as plt

pathImage="2.jpg"
heightImg=450
widthImg=450

#### LOAD THE TRAINNED MODEL
model = load_model('myModel.h5')

# prepare image
img=cv2.imread(pathImage)
img=cv2.resize(img,(widthImg,heightImg))
imgBlank=np.zeros((heightImg,widthImg,3),np.uint8)
imgThreshold=preProcess(img)

#cv2.imshow("Image",imgThreshold)
#cv2.waitKey(0)

# find countours
imgContours=img.copy()
imgBigContours=img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

#cv2.imshow("Image",imgContours)
#cv2.waitKey(0)

#### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
print(biggest)

if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image",imgBigContours)
#cv2.waitKey(0)

# split img
imgSolvedDigits=imgBlank.copy()
boxes=splitBoxes(imgWarpColored)
numbers = getPredection(boxes,model)
print(numbers)
imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))

#cv2.imshow("Image",imgDetectedDigits)
#cv2.waitKey(0)

numbers = np.asarray(numbers)
posArray = np.where(numbers > 0, 0, 1)
print(posArray)

board = np.array_split(numbers, 9)
print(board)
try:
    sudokuSolver.solve(board)
except:
    pass

print(board)
flatList = []
for sublist in board:
    for item in sublist:
        flatList.append(item)

# Display all

imgResult=imgBlank.copy()
imgResult = displayNumbers(imgResult,flatList, color=(255, 0, 255))
#cv2.imshow("ImageResult",imgResult)
#cv2.waitKey(0)

solvedNumbers = flatList * posArray
imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)
#cv2.imshow("ImageResult",imgSolvedDigits)
#cv2.waitKey(0)

# #### 6. OVERLAY SOLUTION
pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER

imgInvWarpColored = img.copy()
imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
#cv2.imshow("ImageInWarp",imgInvWarpColored)
#cv2.waitKey(0)

imgDetectedDigits = drawGrid(imgDetectedDigits)
imgSolvedDigits = drawGrid(imgSolvedDigits)

images=[img,imgThreshold,imgContours,imgBigContours,imgDetectedDigits,imgSolvedDigits,imgInvWarpColored,inv_perspective]
titles=['image','Thresholding','Contours','Big contour','Detected Digits','Solved digits','imgInvWarp','inv_perspective']
siz=len(titles)

for i in range(0,siz):
    plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()


cv2.waitKey(0)