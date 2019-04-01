import nibabel as nib
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import time
import pickle


class ImageData:
    # Contains labelled and image data from a single ct-scan series
    def __init__(self):
        self.label_data = []
        self.image_data = []
        self.image_files = []
        self.label_files = []
        self.num_scans = 0

    # Setters
    def setLabelData(self, _label_data):
        # Set numpy array for labelled data
        self.label_data = _label_data

    def setImageData(self, _image_data):
        # Set numpy array for dicom data
        self.image_data = _image_data

    def setImageFile(self, _image_files):
        # Array of all dicom files
        self.image_files = _image_files

    def setLabelFile(self, _label_files):
        # Array of all label files
        self.label_files = _label_files

    def setNumScans(self, _num_scans):
        # Number of images in this scan series
        # Should be equal to image_data.shape[2] and label_data.shape[2]
        self.num_scans = _num_scans

    # Getters
    def getLabelData(self):
        return self.label_data

    def getImageData(self):
        return self.image_data

    def getImageFile(self):
        return self.image_files

    def getLabelFile(self):
        return self.label_files

    def getNumScans(self):
        return self.num_scans

def removeOuterCircle(dcm_data, image_width=512):
    img_center = (image_width/2, image_width/2)
    print(dcm_data.shape)
    for (index,val) in np.ndenumerate(dcm_data):
        a = index[0]
        b = index[1]
        if sqrt((a-img_center[0])**2+(b-img_center[1])**2) >= img_width/2:
            dcm_data = dcm_data
    time.sleep(500)

def readDCM(dcm_files):
    # Reads dcm files at dcm_files, returns numpy array of dicom scan data
    num_images = 0
    dcm_data=[]
    for dcm_file in dcm_files:
        ds = pydicom.dcmread(dcm_file)
        print(dcm_file)
        try:
            dcm_data.append(ds.pixel_array)
        except:
            print("Error: Skipping scans and labels from "+dcm_file)
            return -1, []
        num_images += 1
        dcm_return = np.asarray(dcm_data)
        #dcm_return[dcm_return > 30000]=0
        #dcm_return = np.sqrt(dcm_data)
        #dcm_return = dcm_return/65536
        mean_value = 12000.0
        #removeOuterCircle(dcm_return,512)
        dcm_mean = np.mean(dcm_return)
        #dcm_return = dcm_return*(mean_value/dcm_mean)

        print("average value: "+str(np.mean(dcm_return)))
    return num_images, dcm_return.astype(np.uint16)#np.asarray(dcm_data)

def readLabel(lbl_files):
    # Reads label files at lbl_files, returns numpy array of label data
    num_images = 0
    lbl_data=[]
    for lbl_file in lbl_files:
        img = nib.load(lbl_file)
        lbl = img.get_fdata()
        lbl = lbl.reshape((lbl.shape[0],lbl.shape[1],lbl.shape[2]))

        # Need to flip label vertically, then rotate counterclock-wise by 270 degrees
        lbl = np.flipud(lbl)
        lbl = np.rot90(lbl, 3,(0,1))
    return lbl.shape[2],lbl.astype(np.uint8)


def loadAllData(all_files):
    # Loads all dcm and label files into an array of ImageData classes
    scan_data = []
    for file_group in all_files:
        dcm_files = file_group[0]
        lbl_files = file_group[1]
        num_dcm,dcm_data = readDCM(dcm_files)
        if num_dcm > 0:
            dcm_data = np.swapaxes(dcm_data, 0,2)
            dcm_data = np.swapaxes(dcm_data, 0,1)

            num_lbl,lbl_data = readLabel(lbl_files)
            if num_lbl != num_dcm:
                print("Error: Mismatch between number of labels and scans. Skipping files at "+lbl_files[0])

            else:
                lbl_data_new = np.zeros(lbl_data.shape)
                i=0
                while True:
                #for i in range(0,lbl_data.shape[2]):
                    if i >= lbl_data.shape[2]:
                        break
                    if 1 not in lbl_data[:,:,i]:
                        num_lbl-=1
                        lbl_data = np.delete(lbl_data, i, axis=2)
                        dcm_data = np.delete(dcm_data, i, axis=2)
                    else:
                        i+=1
                print(lbl_data.shape)

                single_image = ImageData()
                single_image.setLabelData(lbl_data)
                single_image.setImageData(dcm_data)
                single_image.setImageFile(dcm_files)
                single_image.setLabelFile(lbl_files)
                single_image.setNumScans(num_lbl)
                scan_data.append(single_image)
    print("\nFinished reading data\n")
    return scan_data



def getImagePaths(dir):
    # Recursively get all files that contain the dcm and label files for all ct-scans
    # Scans through all subdirectories starting at 'dir'
    global all_files
    full_path = []
    img_path = []
    label_path = []
    img_files = []
    label_files = []
    for file in os.listdir(dir):
        dir_path = dir+'/'+file

        if os.path.isdir(dir_path):
            if file == "dicomm":
                for img_file in sorted(os.listdir(dir_path)):
                    if img_file.endswith(".dcm"):
                        img_files.append(dir_path+'/'+img_file)
                    #print("dicom file: "+dir_path+'/'+img_file)
            elif file == "labels":
                for lbl_file in sorted(os.listdir(dir_path)):
                    if lbl_file.endswith(".hdr") or lbl_file.endswith(".img"):
                        label_files.append(dir_path+'/'+lbl_file)

            else:

                getImagePaths(dir_path)

    if img_files != [] and label_files != []:
        all_files.append([img_files,label_files])

def load_data():
    global all_files
    # data_path is the top directory that contains all label and dicom files
    data_path = "E:/CMPUT466/LabelledData/LabelledData"

    # Array of all label and dcm files
    all_files = []

    # Recursively get all files that contain the dcm and label files for a single scan series
    getImagePaths(data_path)
    ScanData=loadAllData(all_files)
    #with open('/Users/julianstys/Documents/CMPUT466/Stroke-Project/ScanData','wb') as fp:
    #    pickle.dump(ScanData,fp)

    #print(process.get_memory_info()[0])

    print("Number of scans: "+str(len(ScanData)))
    return ScanData
    # Plot ct-scan
    #fig1, ax1 = plt.subplots()
    #ax1.imshow(ScanData[0].getImageData()[:,:,13],cmap='gray')

    # Plot mask
    #fig2, ax2 = plt.subplots()
    #ax2.imshow(ScanData[0].getLabelData()[:,:,13],cmap='gray')
    #plt.pause(5000)
    #time.sleep(5000)



if __name__ == "__main__":
    main()
