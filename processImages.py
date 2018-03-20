import os
from PIL import Image, ImageFilter
import numpy
#import cv2


samplesize = 10000000 #can be used for tesing the process with a smaller sample of images
#FTRAIN = "/home/clab/Downloads/proj/train5/"
FTEST = "/home/cse/Downloads/output/test/"


#outputpath = "/home/cse/ubuntu/pre/train_images_128_128_all_new.csv"
#labelspath ='/home/clab/Downloads/proj/trainLabels5.csv'


def ImagesToFlatFile(outputpath,outputNamesPath,test=False):
    #a converting function that will be used to convert each image into a flatten array
    def convert_image(path):
        im = Image.open(path)
        s = im.resize((128, 128),Image.ANTIALIAS) #resizing image
        s1= s.convert('L') 
        s1=s1.filter(ImageFilter.EDGE_ENHANCE_MORE)
        s1=s1.filter(ImageFilter.FIND_EDGES)
	# convert image to monochrome
	#s1=cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
	#clahe=cv2.createCLAHE(clipLimit=2.0, titleGridSize=(8,8))
	#c1=clache.apply(s1)
        data = numpy.asarray(s1) #converting to an array of bits
        data = data.astype(numpy.float32)/255 #normalizing to values between 0 and 1
        data = data.flatten() #an image is represented as a processed 1D array
        return data 
    i = 0
    inputdirectory = FTEST #if test else FTRAIN
    # Reading csv file that contains image-names and labels
    #labels = numpy.recfromcsv(labelspath, delimiter=',')
    namesfile = open(outputNamesPath, "w")
    #looping over input train images and adding each processed image into the output file    
    with open(outputpath, "w") as outfile: #openning the output file for writing
        for filename in os.listdir(inputdirectory): #looping over each image
            i+=1 #updating number of seen images
            print(i)
            if i >samplesize: break #checking to stop at simulation size limit
            #reading the curent image's label from the labels file
            else:
                fullname = inputdirectory +filename
                label = 9
            #flip right eye's images to the other direction so right and left eyes can be treated with a single model
                if "right" in filename: 
                    X = convert_image(fullname).reshape(128,128)
                    X = numpy.fliplr(X)
                    X = X.reshape(128*128)
                else: 
                    X = convert_image(fullname)
                #write converted data to file
                outfile.write( str(numpy.append(label,convert_image(fullname)).tolist()).strip('[]'))
                outfile.write("\n")
                #write image names to file
                namesfile.write( filename[0:filename.find('.')])
                namesfile.write("\n")
    outfile.close
    namesfile.close

#Writing train images into a file
outputpath = "/home/cse/Downloads/output/test_images_128_128_all_new.csv"
outputNamesPath = "/home/cse/Downloads/output/test_images_128_128_names_new.csv"
ImagesToFlatFile(outputpath,outputNamesPath,False)

