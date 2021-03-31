#读取整个bacepath文件夹下的文件并且转换为8位保存到savepath
import os
import cv2
bacepath = "E:\dataset\8images\8images"
savepath = 'E:\dataset\8images'

f_n  = os.listdir(bacepath)
print(f_n)
for n in f_n:
    imdir = bacepath + '\\' + n
    print(n)
    img = cv2.imread(imdir)

    cropped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(savepath + '\\' + n.split('.')[0] + '_1.png', cropped)  # NOT CAHNGE THE TYPE