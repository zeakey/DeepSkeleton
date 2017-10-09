# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc
import Image
import scipy.io
import os
from os.path import join, splitext
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
# net = caffe.Net('./deploy.prototxt','DS_iter_15000.caffemodel', caffe.TEST)
net = caffe.Net('./deploy.prototxt','sk1491_iter_100000.caffemodel', caffe.TEST)
#test_dir = '/home/server002/zk/data/bsds_parts_imgs/'
#test_dir = '/media/server002/0614-8C40/imgs0.5/'
#save_dir = '//media/server002/0614-8C40/imgs0.5/sk1491/'
#test_dir = '/home/server005/zk/DeepSkeleton/examples/scale-reg1491/sed/2obj_imgs/'
#save_dir = '/home/server005/zk/DeepSkeleton/examples/scale-reg1491/sed/2obj_save/'
test_dir = '/media/data1/zk/TIP2016/sk1491/test/'
save_dir = '/home/zhaokai/gyl/DeepSkeleton/examples/scale-reg1491/result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
imgs = os.listdir(test_dir)
nimgs = len(imgs)
print "totally "+str(nimgs)+"images"
for i in range(nimgs):
    img = imgs[i]
    img = Image.open(test_dir + img)
    img = np.array(img, dtype=np.float32)
    if len(img.shape) == 2:
        img.resize(img.shape[0], img.shape[1], 1)
        img = np.repeat(img, 3, 2)
    img = img[:,:,::-1]
    img -= np.array((104.00698793,116.66876762,122.67891434))
    img = img.transpose((2,0,1))
    net.blobs['data'].reshape(1, *img.shape)
    net.blobs['data'].data[...] = img
    net.forward()
    #out2 = net.blobs['dsn2-out'].data[0][0,:,:]
    #out3 = net.blobs['dsn3-out'].data[0][0,:,:]
    #out4 = net.blobs['dsn4-out'].data[0][0,:,:]
    #out5 = net.blobs['dsn5-out'].data[0][0,:,:]
    #fuse = net.blobs['fuse-out'].data[0][0,:,:]
    concat_reg = net.blobs['concat-reg'].data[0][...]

    full = net.blobs['fuse-out'].data[0][...]
    fuse = full[0, :, :]
    fn, ext = splitext(imgs[i])
    scipy.io.savemat(join(save_dir, fn),dict({'sk':full}),appendmat=True)
    scipy.io.savemat(join(save_dir, fn + '_reg'),dict({'reg':concat_reg}),appendmat=True)    
    
    scipy.misc.imsave(join(save_dir, fn + '.png'), 1-fuse/fuse.max())
    
    print imgs[i]+"   ("+str(i+1)+" of "+str(nimgs)+")    saved"
