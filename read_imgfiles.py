import numpy as np
import os
#import bitstring
import time
#from bitarray import bitarray
import matplotlib.pyplot as plt

class OBJ_data:
    # Contains pixel data from .obj files in a numpy array
    def __init__(self):
        self.obj_data=[]

    # Setters
    def setObjData(self, data):
        self.obj_data = data

    # Getters
    def getObjData(self):
        return self.obj_data

class AVW_data:
    # Contains pixel and header data from .avw files
    # Header data is stored in header_data dictionary
    # Image data is stored in image_data numpy array which has dimmension (width, height, # images)
    def __init__(self):
        # Header and image info
        self.image_num=0
        self.header_data={}
        self.image_data=[]

    # Setters
    def setImageNum(self, num):
        self.image_num = num

    def insertIntoHeader(self, key, value):
        self.header_data[key] = value

    def setImageData(self, np_data):
        self.image_data = np_data

    # Getters
    def getHeaderData(self):
        return self.header_data

    def getImageData(self):
        return self.image_data

    def getHeaderAtKey(self, key):
        return self.header_data[key]

    def getImageNum(self):
        return self.image_num


def readObj(obj_file):
    # Reads data from .obj file, and returns an OBJ_data class
    fp = open(obj_file)
    #a = bitarray()
    #a.fromfile(fp)
    #print(a)
    #print(a.length())
    np_file_data = np.fromfile(fp,dtype=np.uint8) # This is wrong, should read bit by bit
    print(np_file_data.shape)
    obj_data = OBJ_data()
    obj_data.setObjData(np_file_data)
    return obj_data


def readAVW(avw_file):
    # Reads data from avw_file, and returns AVW_data class
    img_info = AVW_data()

    byte_offset=-1
    ### Read header
    index=0
    fp = open(avw_file)
    for line in fp.readlines():
        line_strip = line.strip()
        if line_strip == "EndInformation":
            break
        elif index == 0:
            new_line = [x for x in line_strip.split(' ') if x]
            byte_offset = int(new_line[2])
        else:
            new_line = [x for x in line_strip.split('=') if x]
            if len(new_line) == 2:
                if new_line[0] == "Depth":
                    img_info.setImageNum(int(new_line[1]))
                img_info.insertIntoHeader(new_line[0],new_line[1])
        index += 1
    fp.close()

    ### Read image data
    fp = open(avw_file)
    fp.seek(byte_offset, os.SEEK_SET)
    img_reshape=(int(img_info.getHeaderAtKey('Width')), int(img_info.getHeaderAtKey('Height')), int(img_info.getHeaderAtKey('Depth')))
    np_file_data = np.fromfile(fp,dtype=np.int16)
    print(np_file_data.size)
    np_image_data = np.reshape(np_file_data, img_reshape[::-1]).swapaxes(0,2)
    img_info.setImageData(np_image_data)

    return img_info


def main():
    obj_image = "ICHTest/ICHTest/Object Maps/ICHADAPTII_001_UAH_F_S_AcuteCT.obj"
    avw_image = "ICHTest/ICHTest/Scans/ICHADAPTII_003_UAH_GMM_24hCT.avw"

    img_info=readAVW(avw_image)
    obj_info=readObj(obj_image)
    np.set_printoptions(threshold=np.nan)
    print(obj_info.getObjData())

    plt.ion()
    plt.show()

    for i in range(0,img_info.getImageNum()):
        plt.imshow(img_info.getImageData()[:,:,i])
        plt.draw()
        plt.pause(0.1)




if __name__ == "__main__":
    main()
