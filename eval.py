from net2 import FCN8
import numpy as np
import load_data as ld
import matplotlib.pyplot as plt
import time
from PIL import Image
import sys
import numpngw

def generate_sample_weights(training_data, class_weight_dictionary):
    sample_weights = []
    print(training_data.shape)

    for i in range(training_data.shape[0]):
        ratio = np.count_nonzero(training_data[i,:,:,:] == 1)/np.count_nonzero(training_data[i,:,:,:] == 0)

        sample_weights.append(1+0*ratio*10)

    return np.asarray(sample_weights)


np.set_printoptions(threshold=sys.maxsize)

ScanData = ld.load_data()
print(len(ScanData))
save_path = "E:/CMPUT466/T5-Stoke/save_files16"

num_slices = 1

class_weights = {0:1, 1:50}

# load data
X = []
Y = []
for i in ScanData:
    for j in range(0,i.getNumScans()):
        X.append(i.getImageData()[:,:,j])
        Y.append(i.getLabelData()[:,:,j])

X = np.expand_dims(X,axis=3)
Y = np.expand_dims(Y,axis=3)

#print(X.shape)
#print(X[0,:,:,:].shape)
#image_arr = np.asarray(np.squeeze(X[12,:,:,:],axis=2)).astype(np.uint32)
#print(image_arr.shape)
#im_pred = Image.fromarray(image_arr*50)
#im_pred.save(save_path+"/pred_mask0.png")


#X = [np.zeros((512,512,num_slices)), np.zeros((512,512,num_slices)), np.zeros((512,512,num_slices)),np.zeros((512,512,num_slices))]
#Y = [np.zeros((512,512,1)), np.zeros((512,512,1)), np.zeros((512,512,1)), np.zeros((512,512,1))]
#X, Y = np.array(X) , np.array(Y)

# load model, default input size 512x512
model = FCN8(image_depth=num_slices)
model.summary()

# Split between training and testing data
from sklearn.utils import shuffle
train_rate = 0.95
index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)
index_test  = list(set(range(X.shape[0])) - set(index_train))

# shuffle data for evalulation
X, Y = shuffle(X,Y)
X_train, y_train = X[index_train],Y[index_train]
#X_train=np.squeeze(X_train,axis=3)
#y_train = np.squeeze(y_train,axis=3)
print(y_train.shape)
print(X_train.shape)

X_test, y_test = X[index_test],Y[index_test]
#X_test = np.squeeze(X_test,axis=3)
#y_test = np.squeeze(y_test,axis=3)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# train
from keras import optimizers
sgd = optimizers.SGD(lr=0.008, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#print(y_test.shape)
#for i in range(0,y_test.shape[0]):
    #label_pred = np.asarray(np.squeeze(y_pred[i,:,:])).astype(np.uint32)
#    label_true = np.asarray(np.squeeze(y_test[i,:,:])).astype(np.uint32)

    #im_pred = Image.fromarray(label_pred*50000)
#    im_true = Image.fromarray(label_true*50000)

    #im_pred.save(save_path+"/pred_mask"+str(i)+".png")
#    im_true.save(save_path+"/true_mask"+str(i)+".png")

#print(generate_sample_weights(y_train, class_weights))
#time.sleep(5000)

hist1 = model.fit(X_train,y_train,
                  validation_data=(X_test,y_test),
                  batch_size=10,epochs=100,verbose=1)#,sample_weight=generate_sample_weights(y_train, class_weights))

print("Predicting")
# test
y_pred = model.predict(X_train) # X_test
print(y_pred.shape)
print("-------------")
#print(np.squeeze(y_pred[0,:,:],axis=3))
print("-------------")
#print(y_predi)
y_predi = np.argmax(y_train, axis=3)
y_testi = np.argmax(y_test, axis=3)
print(y_testi.shape,y_predi.shape)
print("Max y_pred: "+str(y_pred.max()))
print("Elements > 0.5: "+str((y_pred>0.5).sum()))
print("Elements <= 0.5: "+str((y_pred<=0.5).sum()))
print("Shape: "+str(y_pred.shape))

#y_pred[y_pred > 0.5] = 1
#y_pred[y_pred <= 0.5] = 0
y_pred = y_pred*50000
y_pred = np.rint(y_pred)
#y_pred = y_pred/y_pred.max()
# https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.htmlx
def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c))
        IoU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))


print(y_predi.shape)
print(y_pred.shape)
print(y_train.shape)
print(X_test.shape)
print(np.squeeze(y_pred[0,:,:]).shape)
if 1 in y_pred:
    print("y_pred has 1s")
else:
    print("y_pred has NO 1s")
non_zero = np.count_nonzero(y_pred)
print("Non zeros: "+str(non_zero))
for i in range(0,y_predi.shape[0]):
    label_pred = np.asarray(np.squeeze(y_pred[i,:,:])).astype(np.uint32)
    label_true = np.asarray(np.squeeze(y_train[i,:,:])).astype(np.uint32)
    image_true = np.asarray(np.squeeze(X_train[i,:,:])).astype(np.uint32)

    im_pred = Image.fromarray(label_pred)
    im_true = Image.fromarray(label_true*50000)
    scan_true = Image.fromarray(image_true*2)

    #scan_true = open(save_path+"/"+str(i)+"_true_img.png")
    #writer = png.Writer(width=512, height=512, bitdepth=16)
    #writer.write(scan_true, image_true)

    np.save(save_path+"/"+str(i)+"_pred_mask.np", label_pred)
    np.save(save_path+"/"+str(i)+"_true_mask.np", label_true)
    np.save(save_path+"/"+str(i)+"_true_img.np", image_true)

    im_pred.save(save_path+"/"+str(i)+"_pred_mask.png")
    im_true.save(save_path+"/"+str(i)+"_true_mask.png")
    scan_true.save(save_path+"/"+str(i)+"_true_img.png")
    ##numpngw.write_png(save_path+"/"+str(i)+"_true_img.jpeg", image_true)

IoU(y_testi,y_predi)
