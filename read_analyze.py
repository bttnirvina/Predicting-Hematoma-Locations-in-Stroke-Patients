import nibabel as nib
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

dataobj = "/Users/julianstys/Documents/CMPUT466/Stroke-Project/LabelledData/ICHADAPT_001_UAH_Z_B/2h/labels/001_2h_ICH-Measurement.img"
hdrobj = "/Users/julianstys/Documents/CMPUT466/Stroke-Project/LabelledData/ICHADAPT_001_UAH_Z_B/2h/labels/001_2h_ICH-Measurement.hdr"
datafile = open(dataobj)
hdrfile = open(hdrobj)
datafile_np = np.fromfile(dataobj)
hdrfile_np = np.fromfile(hdrfile)


img = nib.load(hdrobj)
data = img.get_fdata()
header=img.header

ims=[]
fig = plt.figure()
plt.ion()
for i in range(0,data.shape[2]):

    im=plt.imshow(data[:,:,i,0], cmap='gray')
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=80, blit=False,repeat_delay=1000)
plt.show(ims)
