#!/usr/bin/env python3


import os, glob, math, cv2, time
import numpy as np
from joblib import Parallel, delayed

data_folder = './dataset/imgs'
img_size = 50
sz = (img_size, img_size)

nprocs =4 


def process_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, sz).transpose((2,0,1)).astype('float32') / 255.0
    return img


def read_training_data(): 
    start = time.time()
    
    X_train = []
    Y_train = []
    
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join( data_folder,'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        X_train.extend(Parallel(n_jobs=nprocs)(delayed(process_image)(im_file) for im_file in files))
        Y_train.extend([j]*len(files))
        
    end = time.time() - start
    print("Loading train data: %.2f seconds" % end)
    return X_train, Y_train 


def process_test_image(img_file):
    return process_image(img_file), os.path.basename(img_file)


def read_testing_data(): 
    start = time.time()
    
    X_test    = []
    X_test_id = []
    
    path  = os.path.join( data_folder, 'test', '*.jpg')
    files = glob.glob(path)
    
    results = Parallel(n_jobs=nprocs)(delayed(process_test_image)(im_file) for im_file in files)
    X_test, X_test_id = zip(*results)
    
    end = time.time() - start
    print("Loading test data: %.2f seconds" % end)
    return X_test, X_test_id 


if "__main__" == __name__:
    
    X_train, Y_train  = read_training_data()  
    #X_test, X_test_id = read_testing_data()  

    print( X_train[0] ) 
    print( Y_train ) 
    print( len(X_train) )
    print( len(X_train[0]) ) 
    print( len(Y_train) )  


