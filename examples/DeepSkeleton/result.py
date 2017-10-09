# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc
import Image
import scipy.io
import os
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net('./deploy.prototxt','./sk506_it14k.caffemodel', caffe.TEST)
test_dir = '../../data/sk506/test/'
save_dir = caffe_root+'results/sk506/9k/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
imgs = os.listdir(test_dir)
nimgs = len(imgs)
print "totally "+str(nimgs)+"images"
for i in range(nimgs):
    img = imgs[i]
    img = Image.open(test_dir + img)
    img = np.array(img, dtype=np.float32)
    img = img[:,:,::-1]
    img -= np.array((104.00698793,116.66876762,122.67891434))
    img = img.transpose((2,0,1))
    net.blobs['data'].reshape(1, *img.shape)
    net.blobs['data'].data[...] = img
    net.forward()
    out2 = net.blobs['dsn2-out'].data[0][0,:,:]
    out3 = net.blobs['dsn3-out'].data[0][0,:,:]
    out4 = net.blobs['dsn4-out'].data[0][0,:,:]
    out5 = net.blobs['dsn5-out'].data[0][0,:,:]
    fuse = net.blobs['fuse-out'].data[0][0,:,:]
    scipy.io.savemat(save_dir + imgs[i][0:-4],dict({'sk':1-fuse/fuse.max()}),appendmat=True)
    scipy.misc.imsave(save_dir + imgs[i],1-fuse/fuse.max())
    print imgs[i]+"   ("+str(i+1)+" of "+str(nimgs)+")    saved"
