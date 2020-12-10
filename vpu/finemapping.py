#coding=utf-8
import cv2
import numpy as np
import math
from scipy.ndimage import filters


def fitLine_ransac(pts,zero_add = 0 ):
    if len(pts)>=2:
        [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((136- x) * vy / vx) + y)
        return lefty+30+zero_add,righty+30+zero_add
    return 0,0



#精定位算法
def findContoursAndDrawBoundingBox(image_rgb):


    line_upper  = [];
    line_lower = [];

    line_experiment = []
    grouped_rects = []
    gray_image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)

    # for k in np.linspace(-1.5, -0.2,10):
    for k in np.linspace(-80, 0, 15):

        # thresh_niblack = threshold_niblack(gray_image, window_size=21, k=k)
        # binary_niblack = gray_image > thresh_niblack
        # binary_niblack = binary_niblack.astype(np.uint8) * 255

        binary_niblack = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,k)
       # cv2.imshow("image1",binary_niblack)
       # cv2.waitKey(0)
#        imagex, contours, hierarchy = cv2.findContours(binary_niblack.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(binary_niblack.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

      #  image_rgb2 = image_rgb.copy()
      #  cv2.drawContours(image_rgb2, contours, -1, (0, 0, 255), 2)
      #  cv2.imshow("img", image_rgb2)
      #  cv2.waitKey(0)

        for contour in contours:
            bdbox = cv2.boundingRect(contour)
            if (bdbox[3]/float(bdbox[2])>0.7 and bdbox[3]*bdbox[2]>100 and bdbox[3]*bdbox[2]<1200) or (bdbox[3]/float(bdbox[2])>3 and bdbox[3]*bdbox[2]<100):
                # cv2.rectangle(rgb,(bdbox[0],bdbox[1]),(bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]),(255,0,0),1)
                line_upper.append([bdbox[0],bdbox[1]])
                line_lower.append([bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]])

                line_experiment.append([bdbox[0],bdbox[1]])
                line_experiment.append([bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]])
                # grouped_rects.append(bdbox)

    rgb = cv2.copyMakeBorder(image_rgb,30,30,0,0,cv2.BORDER_REPLICATE)
    leftyA, rightyA = fitLine_ransac(np.array(line_lower),3)
    rows,cols = rgb.shape[:2]

    # rgb = cv2.line(rgb, (cols - 1, rightyA), (0, leftyA), (0, 0, 255), 1,cv2.LINE_AA)

    leftyB, rightyB = fitLine_ransac(np.array(line_upper),-3)

    rows,cols = rgb.shape[:2]

    # rgb = cv2.line(rgb, (cols - 1, rightyB), (0, leftyB), (0,255, 0), 1,cv2.LINE_AA)
    pts_map1  = np.float32([[cols - 1, rightyA], [0, leftyA],[cols - 1, rightyB], [0, leftyB]])
    pts_map2 = np.float32([[136,36],[0,36],[136,0],[0,0]])
    mat = cv2.getPerspectiveTransform(pts_map1,pts_map2)
    image = cv2.warpPerspective(rgb,mat,(136,36),flags=cv2.INTER_CUBIC)

    # cv2.imshow("img", image)
    # cv2.waitKey(0)

    image,M = fastDeskew(image)

    return image

def angle(x,y):
    return int(math.atan2(float(y),float(x))*180.0/3.1415)

def v_rot(img, angel, shape, max_angel):
    size_o = [shape[1],shape[0]]
    size = (shape[1]+ int(shape[0]*np.cos((float(max_angel )/180) * 3.14)),shape[0])
    interval = abs( int( np.sin((float(angel) /180) * 3.14)* shape[0]))
    pts1 = np.float32([[0,0],[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size)
    return dst,M

def skew_detection(image_gray):
    h, w = image_gray.shape[:2]
    eigen = cv2.cornerEigenValsAndVecs(image_gray,12, 5)
    angle_sur = np.zeros(180,np.uint)
    eigen = eigen.reshape(h, w, 3, 2)
    flow = eigen[:,:,2]
    vis = image_gray.copy()
    vis[:] = (192 + np.uint32(vis)) / 2
    d = 12
    points =  np.dstack( np.mgrid[d/2:w:d, d/2:h:d] ).reshape(-1, 2)
    for x, y in points:
        vx, vy = np.int32(flow[int(y), int(x)]*d)
        # cv2.line(rgb, (x-vx, y-vy), (x+vx, y+vy), (0, 355, 0), 1, cv2.LINE_AA)
        ang = angle(vx,vy)
        angle_sur[(ang+180)%180] +=1

    # torr_bin = 30
    angle_sur = angle_sur.astype(np.float)
    angle_sur = (angle_sur-angle_sur.min())/(angle_sur.max()-angle_sur.min())
    angle_sur = filters.gaussian_filter1d(angle_sur,5)
    skew_v_val =  angle_sur[20:180-20].max()
    skew_v = angle_sur[30:180-30].argmax() + 30
    skew_h_A = angle_sur[0:30].max()
    skew_h_B = angle_sur[150:180].max()
    skew_h = 0
    if (skew_h_A > skew_v_val*0.3 or skew_h_B > skew_v_val*0.3):
        if skew_h_A>=skew_h_B:
            skew_h = angle_sur[0:20].argmax()
        else:
            skew_h = - angle_sur[160:180].argmax()
    return skew_h,skew_v


def fastDeskew(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    skew_h,skew_v = skew_detection(image_gray)
    # print("校正角度 h ",skew_h,"v",skew_v)
    deskew,M = v_rot(image,int((90-skew_v)*1.5),image.shape,60)
    return deskew,M
