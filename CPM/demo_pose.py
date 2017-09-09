import cv2 as cv
import numpy as np




img_dir='/data/COCO/2014_2015/train2014/'
img_color =cv.imread(img_dir+'COCO_train2014_000000000009.jpg', cv.IMREAD_COLOR)

h,w,c= img_color.shape
print('height ',h,'\n wieght:',w,'\ncolor:',c)
img_gray=np.zeros(shape=(h,w),dtype=np.uint8)

for y in range(0,h):
    for x in range(0,w):
        b=img_color[y,x,0]
        g=img_color[y,x,1]
        r=img_color[y,x,2]

        gray=(int(b)+int(g)+int(r))/3.0

        if gray >255:
            gray=255
        img_gray[y,x]=int(gray)

cv.imshow('Color image',img_color)
cv.imshow('Gray_image',img_gray)

cv.imwrite('result.jpg',img_gray)

cv.waitKey(0)
cv.destroyAllWindows()


