# -*- coding: UTF-8 -*-

import sys
import dlib
import cv2
import io
import urllib
import numpy as np
import json
import os
from scipy import interpolate
from scipy.interpolate import interp1d
from pylab import *
from skimage import color
from PIL import Image

texture_input = 'texture1.jpg'
text = imread(texture_input)

def showImg(imOrg1):
    plt.imshow(imOrg1)
    imsave('output.jpg', imOrg1)
    show()

def faces(im):
        b, g, r = cv2.split(im)    #Spliting the image feed by the three basic color
        im = cv2.merge([r, g, b])
        #print 'printing from faces'
        #imshow(im)
        #show()
        detector = dlib.get_frontal_face_detector()
        dets = detector(im, 1) #Use the detctor model
        print("Number of faces detected: {}".format(len(dets)))  #number of human faces recognized
        if dets:
            #for index, face in enumerate(dets):
            return im,dets

def landmarks(im,face):
    print 'printing from landmarks'
    im=im.copy()
    current_path = os.getcwd()
    predictor_path = current_path + "//model//2.dat"
    predictor = dlib.shape_predictor(predictor_path)
    shape = predictor(im, face)
    list1 = []
    coords = []
    for index, pt in enumerate(shape.parts()):
        list1.append('Part{}: {}'.format(index,pt))
        pt_pos = (pt.x, pt.y)
        coords.append ((int(pt.x),int(pt.y)))
        #cv2.circle(im, pt_pos, 2, (255, 255, 0), 1)
        #print type(pt),type(pt.x),type(int(pt.y)),type(pt_pos)
        pointox,pointoy,pointix,pointiy=[],[],[],[]
        #Locking up coords for the positioning of the mouth
    for i in range(48,60):
        pointox.append(coords[i][0])
        pointoy.append(coords[i][1])
    for i in range(60,68):
        pointix.append(coords[i][0])
        pointiy.append(coords[i][1])
    pointox = np.array(pointox)
    pointoy = np.array(pointoy)
    pointix = np.array(pointix)
    pointiy = np.array(pointiy)
    text1=json.dumps(list1)
    return pointox,pointoy,pointix,pointiy,im

def lips(point_out_x,point_out_y,point_in_x,point_in_y,im):
    r, g, b = (245., 150., 170.)  # lipstick color
    up_left_end = 4
    up_right_end = 7

    def inter(lx, ly, k1='quadratic'):
        unew = np.arange(lx[0], lx[-1] + 1, 1) #找到数组的区间？
        f2 = interp1d(lx, ly, kind=k1) #利用差值来连线？
        return f2, unew
    # Code for the curves bounding the lips
    o_u_l = inter(point_out_x[:up_left_end], point_out_y[:up_left_end])
    o_u_r = inter(point_out_x[up_left_end - 1:up_right_end], point_out_y[up_left_end - 1:up_right_end])
    o_l = inter([point_out_x[0]] + point_out_x[up_right_end - 1:][::-1].tolist(),
                [point_out_y[0]] + point_out_y[up_right_end - 1:][::-1].tolist(), 'cubic')
    i_u_l = inter(point_in_x[:3], point_in_y[:3])
    i_u_r = inter(point_in_x[3 - 1:5], point_in_y[3 - 1:5])
    i_l = inter([point_in_x[0]] + point_in_x[5 - 1:][::-1].tolist(),
                [point_in_y[0]] + point_in_y[5 - 1:][::-1].tolist(), 'cubic')
    x = []  # will contain the x coordinates of points on lips
    y = []  # will contain the y coordinates of points on lips

    def ext(a, b, i):
        a, b = np.round(a), np.round(b)
        x.extend(arange(a, b, 1, dtype=np.int32).tolist())
        y.extend((np.ones(int(b - a), dtype=np.int32) * i).tolist())

    for i in range(int(o_u_l[1][0]), int(i_u_l[1][0] + 1)):
        ext(o_u_l[0](i), o_l[0](i) + 1, i)
    for i in range(int(i_u_l[1][0]), int(o_u_l[1][-1] + 1)):
        ext(o_u_l[0](i), i_u_l[0](i) + 1, i)
        ext(i_l[0](i), o_l[0](i) + 1, i)

    for i in range(int(i_u_r[1][-1]), int(o_u_r[1][-1] + 1)):
        ext(o_u_r[0](i), o_l[0](i) + 1, i)
    for i in range(int(i_u_r[1][0]), int(i_u_r[1][-1] + 1)):
        ext(o_u_r[0](i), i_u_r[0](i) + 1, i)
        ext(i_l[0](i), o_l[0](i) + 1, i)
    # Now x and y contains coordinates of all the points on lips

    #Coverting the region marked by array x and y from RGB to LAB
    val = color.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
    #Find the average brightness and a, b value of the region
    L, A, B = mean(val[:, 0]), mean(val[:, 1]), mean(val[:, 2])
    #Since we know the color we would like to achieve, convert that to LAB as well
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
    #Find the difference in average
    ll, aa, bb = L1 - L, A1 - A, B1 - B
    #Add the difference to each pixel so that the brightness and contrast statys the same
    val[:, 0] += ll
    val[:, 1] += aa
    val[:, 2] += bb
    im[x, y] = color.lab2rgb(val.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255
    gca().set_aspect('equal', adjustable='box')


    #imshow(im)
    #show()
    xmin, ymin = amin(x), amin(y)
    print 'xmin',xmin
    print 'ymin',ymin
    '''
    txt = Image.open(texture_input)
    pix = txt.load()
    print 'size',txt.size  # Get the width and hight of the image for iterating over
    print 'pix',pix[50,50]  # Get the RGBA Value of the a pixel of an image
    #pix[50,50] = value  # Set the RGBA Value of the image (tuple)
    #print 'value',value
    txt.save('alive_parrot.png')  # Save the modified pixels as .png
    '''
    X = (x - xmin).astype(int)
    Y = (y - ymin).astype(int)
    print 'X',X
    print 'Y',Y
    val1 = color.rgb2lab((text[X, Y] / 255.).reshape(len(X), 1, 3)).reshape(len(X), 3)
    #val1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
    val2 = color.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
    #print 'val1 val2 ',val1,val2
    L, A, B = mean(val2[:, 0]), mean(val2[:, 1]), mean(val2[:, 2])
    imshow(im)
    show()
    '''
    val2[:, 0] = np.clip(val2[:, 0] - L + val1[:, 0], 0, 100)
    val2[:, 1] = np.clip(val2[:, 1] - A + val1[:, 1], -127, 128)
    val2[:, 2] = np.clip(val2[:, 2] - B + val1[:, 2], -127, 128) +(100-max(val2[:,0]))
    '''
    val2[:, 0] = np.clip(0.4*val2[:, 0]+(val2[:, 0]-val1[:, 0]), 0, 100)
    #val2[:, 0] = np.clip(0.5*(val2[:, 0] + val1[:, 0]), 0, 100)
    #val2[:, 1] = np.clip(val2[:, 1] - A + val1[:, 1], -127, 128)
    #val2[:, 2] = np.clip(val2[:, 2] - B + val1[:, 2], -127, 128)

    im[x, y] = color.lab2rgb(val2.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255
    return im




for f in sys.argv[1:]:
    f='pics/'+str(f)
    str1 = f
    print str1
    if 'http' in str1:
        cap = cv2.VideoCapture(f)
        ret,img = cap.read()
        img = cv2.imread(im, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(f, cv2.IMREAD_COLOR)


    im=img.copy()
    print type(im)
    im,dets=faces(im)
    for i,fac in enumerate(dets):
        try:
            #print 'try---------------------------------------'
            pointox,pointoy,pointix,pointiy,im = landmarks(im,fac)
            #print 'try',pointox,pointoy,pointix,pointiy
            im=lips(pointox,pointoy,pointix,pointiy,im)
        except ValueError:
            im = cv2.flip(im, 1)
            b, g, r = cv2.split(im)    #Spliting the image feed by the three basic color
            im = cv2.merge([r, g, b])
            print 'Exception encountered'
            #im = im[top:bot, left:right]
            #cv2.imwrite("a.jpeg",im)
            #im = cv2.imread(f, cv2.IMREAD_COLOR)
            print 'should be flipped',type(im)
            im,dets1=faces(im)
            j=i
            for k,face in enumerate(dets1):
                if k==j:
                    pointox,pointoy,pointix,pointiy,im = landmarks(im,face)
                    print 'except',pointox,pointoy,pointix,pointiy
                    lips(pointox,pointoy,pointix,pointiy,im)
            im = cv2.flip(im, 1)



imOrg = im.copy()
showImg(imOrg)
