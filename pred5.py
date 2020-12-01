import sys
#from tensorflow import keras 
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
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


def pred1(img_path):     
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import sys, cv2, random   
    
    
    color_type = 1  
    img_rows, img_cols = 256, 256   
   
    def get_im_cv2(path, img_rows, img_cols, color_type=1):
        # Load as grayscale
        if color_type == 1:
            img = cv2.imread(path, 0)
        elif color_type == 3:
            img = cv2.imread(path)
        # Reduce size
        #resized = cv2.resize(img, (img_cols, img_rows))
        # Keep size
        resized =  img
        print("img size: ", len(img), len(img[0]) )
        return resized
    
    def get_im_cv2_mod(path, img_rows, img_cols, color_type=1):
        # Load as grayscale
        if color_type == 1:
            img = cv2.imread(path, 0)
        else:
            img = cv2.imread(path)
        # Reduce size
        rotate = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
        return resized
    
    model = keras.models.load_model('trained_model/savedmodel.pb')
    img_ori = cv2.imread( sys.argv[1] ) 
    print( len(img_ori), len(img_ori[0]) ) 
    print( img_ori ) 
    img_shrunk = get_im_cv2_mod( sys.argv[1], img_rows, img_cols, color_type)   
    print( len(img_shrunk), len(img_shrunk[0]) ) 
    print( img_shrunk ) 
    
    
    result = model.predict(img_ori)
    print("Prediction is: ", result)  
    

def preprocessing_ori( img_path ):
  
    filename = img_path       
    # load an image in PIL format
    original_image = load_img(filename, target_size=(256, 256))
    
    # convert the PIL image (width, height) to a NumPy array (height, width, channel)
    numpy_image = img_to_array(original_image) 
    
    # Convert the image into 4D Tensor (samples, height, width, channels) by adding an extra dimension to the axis 0.
    input_image = np.expand_dims(numpy_image, axis=0)
    
    print('PIL image size = ', original_image.size)
    print('NumPy image size = ', numpy_image.shape)
    print('Input image size = ', input_image.shape)
    #plt.imshow(np.uint8(input_image[0]))
    
    return input_image  

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
    img_reshaped = gray_image.reshape( 1, 1, IMG_SIZE, IMG_SIZE)

    return img_reshaped   

def preprocessing( img_path ):
    IMG_SIZE = 256  # 50 in txt-based
    img_array = cv2.imread(img_path)  # read in the image, convert to grayscale
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(1, 1, IMG_SIZE, IMG_SIZE )  # return the image with shaping that TF wants.


def pred2(img_path):     
    
    input_image = prepro_2( img_path )   
   
    
    #preprocess for vgg16
    #processed_image_vgg16 = vgg16.preprocess_input(input_image.copy())
    custom3 = keras.models.load_model('trained_model/trained5.h5')
    #processed_image_custom3 = custom3.preprocess_input(input_image.copy())
    processed_image_custom3 = custom3  
    # model  
    predictions_model = processed_image_custom3.predict_classes( input_image )
    #label_custom3 = decode_predictions(predictions_model)
    #print ('label_custom3 = ', label_custom3)
    print( 'prediction is: ', predictions_model  ) 
    print( 'Predicted label: C', predictions_model[0], ',', label_set[ predictions_model[0]] ) 

if __name__ == "__main__":
    img_path = sys.argv[1] 
    #pred1(img_path) 
    pred2(img_path) 



