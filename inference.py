import sys
from tensorflow import keras
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from tensorflow.keras.applications.imagenet_utils import preprocess_input
#import matplotlib.pyplot as plt 
import numpy as np
import sys, cv2   

label_set = [ 
             'normal driving',                        # C0
             'texting - right',                       # C1
             'talking on the phone - right',          # C2
             'texting - left',                        # C3
             'talking on the phone - left',           # C4
             'operating the radio',                   # C5
             'drinking',                              # C6  
             'reaching behind',                       # C7 
             'hair and makeup',                       # C8  
             'talking to passenger'                   # C9 
             ]    

 

def prepro_2( img_path ):
    IMG_SIZE = 256  # 50 in txt-based
    img_array = cv2.imread(img_path)  # read in the image, convert to grayscale
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    gray_image = np.array(gray_image, dtype=np.float32)
    gray_image /= 255
    #train_target = np_utils.to_categorical(train_target, 10) 
    #print('100th y shape:', train_target[66])  
    #img_fl = img_unit8.astype('float32')
    img_reshaped = gray_image.reshape( 1, IMG_SIZE, IMG_SIZE, 1)   

    return img_reshaped   


def pred2(img_path):     
    
    input_image = prepro_2( img_path )   
   
    
    #preprocess for vgg16 
    #processed_image_vgg16 = vgg16.preprocess_input(input_image.copy()) 
    custom3 = keras.models.load_model('trained_model/trained5.h5') 
    #processed_image_custom3 = custom3.preprocess_input(input_image.copy()) 
    processed_image_custom3 = custom3  
    # model  c
    predictions_model = processed_image_custom3.predict_classes( input_image ) 
    #label_custom3 = decode_predictions(predictions_model)
    #print ('label_custom3 = ', label_custom3)
    #print( 'prediction is: ', predictions_model  ) 
    print( 'Predicted label: C', predictions_model[0], ',', label_set[ predictions_model[0]] ) 

if __name__ == "__main__":
    img_path = sys.argv[1] 
    #pred1(img_path) 
    pred2(img_path) 



