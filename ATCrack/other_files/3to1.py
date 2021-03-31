from PIL import Image
import os
path = 'E:\dataset\8images\8images'
file_list = os.listdir(path)
for file in file_list:
    I = Image.open(path+"/"+file)
    L = I.convert('L')
    L.save(path+"/"+file)
    #print(file)