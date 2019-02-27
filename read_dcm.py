import matplotlib.pyplot as plt
import os, time
import pydicom
from pydicom.data import get_testdata_files

dir_name = "ICHTest/ICHTest/Patient Example 2/Scan1/ICHADAPT_015_UAH_JAB_19480810_91077945/"
plt.ion()
plt.show()
for filename in sorted(os.listdir(dir_name)):
    ds = pydicom.dcmread(dir_name + filename)
    print(filename)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.draw()
    plt.pause(0.01)
