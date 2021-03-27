import cv2
import numpy as np
import pickle


grid=np.array([])

def preprocessing(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur=cv2.GaussianBlur(img_gray,(5,5),1)
    img_thresh=cv2.adaptiveThreshold(img_blur,255,1,1,11,2)
    return img_thresh

def findcontours(img):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def order_points(pts):
    pts=pts.reshape((4,2))
    new_points=np.zeros((4,1,2),dtype=np.int32)
    sum=np.sum(pts,axis=1)
    new_points[0]=pts[np.argmin(sum)]
    new_points[3]=pts[np.argmax(sum)]
    diff=np.diff(pts,axis=1)
    new_points[1]=pts[np.argmin(diff)]
    new_points[2]=pts[np.argmax(diff)]
    return new_points

def  transform_image(pts,img3):
    pts1 = np.float32(pts)
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])

    t_matrix= cv2.getPerspectiveTransform(pts1, pts2)
    print(t_matrix.shape)

    dst = cv2.warpPerspective(img3, t_matrix, (450,450))
    return dst

def split_image(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes


def predictions(boxes,model):

    result=[]
    for image in boxes:
        img=image[7:image.shape[0]-7,6:image.shape[1]-6]
        img=cv2.resize(img,(32,32))
        img=img/255
        img=img.reshape(1,32,32,1)
        prediction=model.predict(img)
        classindex=model.predict_classes(img)
        probabilityValue = np.amax(prediction)
        if probabilityValue > 0.7:
            result.append(classindex[0])
        else:
            result.append(0)
    return result

def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),  (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,2, color, 2, cv2.LINE_AA)
    return img


def solve(grid):
    find = find_empty(grid)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1,10):
        if valid(grid, i, (row, col)):
            grid[row][col] = i
            if solve(grid):
                return True
            grid[row][col] = 0
    return False

def valid(grid, num, pos):
    for i in range(len(grid[0])):
        if grid[pos[0]][i] == num:
            return False
    for i in range(len(grid)):
        if grid[i][pos[1]] == num :
            return False
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if grid[i][j] == num:
                return False
    return True

def find_empty(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                return (i, j)
    return None
cap = cv2.VideoCapture(0)

while True:
    ret, fram = cap.read()
    img3=fram.copy()
    print(fram.shape)
    imgBlank = np.zeros((480, 640, 3), np.uint8)  #camera
    imgBlank1=np.zeros((450, 450, 3), np.uint8)
    imgDetectedDigits = imgBlank1.copy()
    img_contour=fram.copy()
    img=preprocessing(img3)
    biggest = findcontours(img)
    if biggest.size != 0:
        img_contour = cv2.drawContours(img_contour, biggest, -1, (0, 0, 255), 5)
        new_points=order_points(biggest)
        wrap_image= transform_image(new_points,img3)
        wrap_image=cv2.cvtColor(wrap_image,cv2.COLOR_BGR2GRAY)
        boxes=split_image(wrap_image)
        pickle_in = open("model_trained_10.p", "rb")
        model = pickle.load(pickle_in)
        numbers = predictions(boxes, model)
        print(numbers)
        imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)
        grid = np.array_split(numbers, 9)
        solve(grid)
        list = []

        for sublist in grid:
            for item in sublist:
                list.append(item)
        solvedNumbers = list * posArray
        imgSolvedDigits = imgBlank1.copy()
        imgSolvedDigits1 = imgBlank.copy()
        imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)
        pts1 = np.float32(new_points)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])

        t_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst	=cv2.warpPerspective(src=imgSolvedDigits,M=t_matrix,dsize=(640,480), flags=cv2.WARP_INVERSE_MAP)

        inv_perspective = cv2.addWeighted(dst, 1, fram, 0.5, 1)

        cv2.imshow('solved_image',  inv_perspective)

    cv2.imshow("frame",fram)
    if cv2.waitKey(0) and 0xFF == ord('q'):
        break






