'''
Image similarity of comprehensive two-dimensional gas chromatographic ( GC×GC ) fingerprints
By Xinyue Yang from Xiangtan University
'''
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
from PIL import ImageFont,ImageDraw, ImageTk,Image

def func_1():
    global img1_gray
    file_path1 = filedialog.askopenfilename()
    img1_gray = cv2.imread(file_path1)
    print('Open file 1：', file_path1)
def func_2():
    global img2_gray
    file_path2= filedialog.askopenfilename()
    print('Open file 2：', file_path2)
    img2_gray = cv2.imread(file_path2)
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(img1_gray, None)
    kp2, des2 = surf.detectAndCompute(img2_gray, None)
    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)
    print(len(des1))
    print(len(des2))
    goodMatch = []
    #Ratio can be adjusted
    ratio=0.98
    for m, n in matches:
        if m.distance < ratio * n.distance:
            goodMatch.append(m)
    if len(des1) < len(des2):
        MAX = len(des2)
        MIN = len(des1)
    else:
        MAX = len(des1)
        MIN = len(des2)
    F=len(goodMatch)
    #Calculate the image similarity of fingerprints
    global r
    if F<MIN:
       r = F / MAX
    else:
       r = MIN / MAX
    print("The feature matching quantity is",F)
    print('The similarity of GC×GC fingerprints is',str('%.4f' % r))
    drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:])
    l5 = tk.Label(window1)
    l5.place(x=250, y=250)
    l5.config(text=' The similarity of GC × GC fingerprints is  ' + str('%.4f' % r) + ' .')
def drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))
    cv2.imwrite("match.png",vis)
    im=cv2.imread("match.png")
    im2=cv2.resize(im,(1500,550))
    cv2.imwrite("match.png", im2)
    match_image = ImageTk.PhotoImage(file='match.png', master=window1)
    imglabel = tk.Label(window1)
    imglabel.place(x=50, y=300)
    imglabel.config(image=match_image )
    imglabel.image = match_image

if __name__ == '__main__':
    window1 = ttk.Window()
    style = ttk.Style(theme='superhero')
    base = style.master
    window1.title('Image similarity of GC×GC fingerprints')
    window1.geometry('1600x900')
    l1 = tk.Label(window1, font=("Times", 13, "bold"), text='Image similarity of GC×GC fingerprints')
    l1.place(x=450, y=15)
    l2 = tk.Label(window1, text='Input 1 :', font=("Times", 11))
    l2.place(x=500, y=80)
    b1 = tk.Button(window1, text="queryImage", width=20, font=("Times", 11), command=func_1)
    b1.place(x=650, y=80)
    l3 = tk.Label(window1, text='Input 2 :', font=("Times", 11))
    l3.place(x=500, y=150)
    b2 = tk.Button(window1, text="referenceImage", width=20, font=("Times", 11), command=func_2)
    b2.place(x=650, y=150)
    l4 = tk.Label(window1, text='Match_result :', font=("Times", 12))
    l4.place(x=50, y=250)
    window1.mainloop()
