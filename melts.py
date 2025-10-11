
import os
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Depends
from fastapi import Header
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PIL import Image
from pydantic import BaseModel
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
from PIL import Image

from typing import Optional
from fastapi import Header
# On Python 3.10+, you could use:
# from fastapi import Header
# from typing import Annotated
# from typing_extensions import Annotated # for older python versions


# --- Globals and Initialization ---
MODEL_PATH = "models/modelMelts.keras"
DATA_PATH = "data/training_data.csv"

# The main FastAPI application instance
app = FastAPI(
    title="TensorFlow Melts ML API",
    description="A class-based REST API for managing a TensorFlow-based Melts model.",
)

# API router for versioning and organization
router = APIRouter(prefix="/v1")

import tensorflow as tf
from mpl_toolkits import mplot3d
import numpy as np
 
#https://www.youtube.com/watch?v=FEchiC1PZ9s&list=PLp-0K3kfddPxRmjgjm0P1WT6H-gTqE8j9&index=22

## Import Necessary Modules
import os
os.environ['TF_USE_LEGACY_KERAS']='1'
import sys
sys.path.append("/home/mahesh/TENSORFLOW/tensorflow/tensorflow/")

import matplotlib.pyplot as plt
import tensorflow.keras as keras
#from tensorflow.python.keras.layers import Conv2D
from tensorflow.keras.layers import Dense,Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
#from tensorflow.keras.datasets import cifar10

from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

import numpy as np
import os

#import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model



# --- Pydantic Response Model ---
# This create struct of string or other paramtere.. and use PredictionResponse as parameter
class PredictionResponse(BaseModel):
    prediction: str



class  melts(Model):

    def __init__(self,inputshape,depth):
        super().__init__()
        self.input_shape = inputshape
        self.depth       = depth
        self.mmodel = self.resnet_v1(self.input_shape,int(self.depth))

        print("constructor or init")

    def getlog(self, logfile:str):
        # --- Logging Configuration ---
        # You can customize this logging config based on your needs.
        # This setup writes logs to a file named 'app.log'.
        logging.basicConfig(
                            level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[
                            logging.FileHandler("meltlogdata.log"),
                            logging.StreamHandler()
                                     ]
                            )
        self.logger = logging.getLogger(__name__)


    def loadModel(self):
        
        if self.mmodel is not None:
            return self.mmodel

        #prexisting model.
        if os.path.exists(MODEL_PATH):
            print("loading the model")
            self.mmodel=load_model(MODEL_PATH)


    def getHistory(self):

        history=self.history
        # Fit the model
        #history = AiModel.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

 # Save the plot to an in-memory buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()  # Close the plot to free up memory

        # Return the buffer content as a StreamingResponse
        return buffer


    def lr_schedule(epoch):
        """Learning Rate Schedule
         Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
         # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def mish(self, inputs):
        x = tf.keras.activations.softmax(inputs)
        x = tf.keras.activations.tanh(x)
        #x = inputs
        #x = tf.math.multiply(x, inputs)
        return x

    def resnet_layer(self,
                     inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation = "NULL",
                     batch_normalization=True,
                     conv_first=True):
   
        """
                2D Convolution-Batch Normalization-Activation stack builder
                Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)
                Returns
                x (tensor): tensor as input to the next layer
        """

        # lass 'keras.src.layers.convolutional.conv2d.Conv2D'>
        print("before conv", num_filters)

        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))


        print(type(conv))

        #breakpoint() 

        print("afeter conv", inputs.shape)
        #batch, with.hght
   
        #y = tf.expand_dims(inputs, axis=3)

        #print("y shape", y.shape)
        x = inputs 
        if conv_first:
            #removing batch dimension, and input is array of 28x28 image
            print("x first", x)
        
            #m = (0,) + inputs
            print("x first0", x.shape)
            #data = [ d for x1 in x  for x2 in x1  for x3 in x2]
  
            #

            x = conv(x)
       
            print("x firs1t", x)

            if batch_normalization:
                x = BatchNormalization()(x)
                print("x firs2t", x.shape)

            if activation is not None:
                x = self.mish(x)
        else:
            x = inputs
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = self.mish(x)
            x = conv(x)
    
        return x


    def On_epoch_begin(self):    
        Print("START TRAINING")

       
    #end of all training records all the weights
    def On_epoch_end(self, epoch, logs=None):

        for layerIndex , layerName in enumerate(self.mmodel.layer):

            # every layer have 2d weights
            self.weight_dict["weight" + str(layerName)]  = self.mmodel.layer[layerIndex].get_weight()[0]
            self.weight_dict["bias" + str(layerName)]    = self.mmodel.layer[layerIndex].get_weight()[1]
      
        if (epoch==0):

            print("stacking")

        else:  
              print("stacking weigts at epoch " , epoch)
              self.weightStack.append(self.weight_dict) 


    def plot_imshow(self,img):
        plt.figure()
        plt.imshow(img[0], interpolation='nearest')
        plt.colorbar()
        plt.grid(False)
        plt.show()





    def plot_layer_outputs(self,layer):    
        x_max = layer.shape[0]
        y_max = layer.shape[1]
        n     = layer.shape[2]

        L = []
        for i in range(n):
            L.append(np.zeros((x_max, y_max)))

        for i in range(n):
            for x in range(x_max):
                for y in range(y_max):
                    L[i][x][y] = layer[x][y][i]

        for img in L:
            plt.figure()
            plt.imshow(img, interpolation='nearest')



    # rewrite these based on donwload pdf trainin one..
    # overwrite train_step() of keras.model.
    @tf.function
    def train_step(self, traindataSet):
         # Unpack the data. Its structure depends on your model and
         # on what you pass to `fit()`.

            #tf.compat.v1.enable_eager_execution()

            print(" called by model.fit")
            x_train, y_train = traindataSet
             

            """
            for index, layer in enumerate(allLayer):
                print("layer", layer, "shpae", layer.shape)
                #plot.subplot(allLayer[index])
                self.plot_layer_outputs(layer)
                breakpoint()
                self.plot_imshow(layer)
                breakpoint()
                      
            """

            with tf.GradientTape() as tape:
                y_pred =self.mmodel(x_train, training=True)
                #Compute loss
                loss = self.mmodel.compute_loss(y=y_train, y_pred=y_pred)


            tf.print("loss ", loss, "weigts", self.mmodel.trainable_weights)
            trainableWeight=self.mmodel.trainable_weights
            #print("loss", loss, "weight", trainableWeight)
            # Compute gradients
            # how many train varibale in model, of 16 layers:
            # compute graduant of loss wrt weigts  
            gradients = tape.gradient(loss,trainableWeight)
            #optimizer = tf.keras.optimizers.SGD()
            #print("update weights by appling gradients ", gradients, "trainableWeight", trainableWeight)
            #optimizer = keras.optimizers.Adam(learning_rate=1e-3)

            breakpoint()

            self.optimizer.apply(zip(trainableWeight, gradients))
            # Update metrics (includes the metric that tracks the loss)
            for metric in self.metrics:
                if metric.name == "loss":
                    metric.update_state(loss)
                else:
                    metric.update_state(y, y_pred)
            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}


    def resnet_v1(self,input_shape, depth, num_classes=2):
        """
            ResNet Version 1 Model builder [a]
            Stacks of 2 x (3 x 3)
            Last ReLU is after the shortcut connection.
            At the beginning of each stage, the feature map size is halved (downsampled)
            by a convolutional layer with strides=2, while the number of filters is
            doubled. Within each stage, the layers have the same number filters and the
            same number of filters.
            Features maps sizes:
            stage 0: 32x32, 16
            stage 1: 16x16, 32
            stage 2:  8x8,  64
            The Number of parameters is approx the same as Table 6 of [a]:
            ResNet20 0.27M
            ResNet32 0.46M
            ResNet44 0.66M
            ResNet56 0.85M
            ResNet110 1.7M
            # Arguments
                input_shape (tensor): shape of input image tensor
                depth (int): number of core convolutional layers
                num_classes (int): number of classes (CIFAR10 has 10)
            # Returns
            model (Model): Keras model instance
        """

        print("depth", depth)

        # depth should  ..
        if((depth - 2) % 6) != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

        # Start model definition.
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6) # 2 
        inputs = tf.keras.Input(self.input_shape)

        # inputs = Input(shape=input_shape)
        #inputs = tf.reshape(inputs, [0, 32, 32, 3])

        x = self.resnet_layer(inputs,num_filters)

        print("layer1", x.shape);

        # Instantiate the stack of residual units
        for stack in range(stackN):
           
            print(" no of stack ", stack, num_res_blocks)
            for res_block in range(num_res_blocks):

                print(" num_res_blocks", res_block, num_res_blocks)
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    print(" this one ")
                    strides = 2  # downsample
                     
                y = self.resnet_layer(inputs=x,
                                      num_filters=num_filters,
                                      strides=strides)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters,
                                      activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    print("this 2")
                    x = self.resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)


                print("ading tow layers")

                x = keras.layers.add([x, y])
                x = self.mish(x)
            num_filters *= 2

        print("reent x:", x, "num filter", num_filters)

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)

        outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)

        model.summary()

        return model


    def call(self, input1):
        #Create models , this is uncompiled model

        # Instantiate model. call init function and instaitaite inputs an doutputs
        #model = Model(inputs=inputs, outputs=self.outputs)

        #self.outputs = self.resnet_v1(self.input_shape,int(self.depth))
 
        # Instantiate model.
        #self.mmodel = Model(inputs=inputs, outputs=self.outputs)

        print("calling....")
        self.mmodel.summary()
      
        #Read data base and pre

        return 0 

        #The CIFAR-10 database is a well-known, large dataset of images used to train and
        #test machine learning and computer vision algorithms. The goal is typically to create a model that
        #can accurately classify the images into their correct categories.
    def getDataset(self): 

        if os.path.exists(DATA_PATH):
            df     = pd.read_csv(DATA_PATH)
            x_data = np.random.rand(len(df),28,28,3).astype(np.float32)
            y_data = pd.get_dumies(df['label']).values
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data,y_data, test_size=0.2,random_state=42)
        else:
            cifara10 = keras.datasets.cifara10
            (self.x_train, self.y_train),(self.x_test,self.y_test) = cifara10.load()
        


    #Let’s view the weights as a 28x28 grid where the weights are arranged exactly like their corresponding pixels.
    def plot_weights(self,shape, weights):
        # Get the values for the weights from the TensorFlow variable.
        w = weights
        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(w)
        w_max = np.max(w)
        print(w_min)
        print(w_max)
        # Create figure with 3x4 sub-plots,
        # where the last 2 sub-plots are unused.
        fig, axes = plt.subplots(3, 4)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Only use the weights for the first 10 sub-plots.
            if i<10:
                # Get the weights for the i'th digit and reshape it.
                # Note that w.shape == (img_size_flat, 10)
                image = w[:, i].reshape(shape)

                # Set the label for the sub-plot.
                ax.set_xlabel("Weights: {0}".format(i))

                # Plot the image.
                ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

            # Remove ticks from each sub-plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
    plt.show()


    #https://www.datacamp.com/tutorial/forward-propagation-neural-networks
    #[‘apple’, ‘orange’, ‘cherry’], the goal is to combine them into a list of tuples like [(1, ‘apple’), (2, ‘orange’), (3, ‘cherry’)].
    #Using zip()
    #zip() is the most efficient approach to combine two or more separate lists into a list of tuples.
    def  ComputeforwardPath(self,inputs):

         layers = self.mmodel.layers
         v = tf.Variable([2.0,2.0,2.0,2.0])
         #w = tf.reshape(v, shape=(4,4))
         print(layers) 
         PrvActivation = inputs  # this is input and for every next layer activation is input 
        
         print("input" , PrvActivation, "shape", inputs.shape)


         #initial_model = keras.Sequential(
         #[
         #   keras.Input(shape=(250, 250, 3)),
         #   layers.Conv2D(32, 5, strides=2, activation="relu"),
         #   layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
         #   layers.Conv2D(32, 3, activation="relu"),
         #]
         #)
         #feature_extractor = keras.Model(
         #inputs=initial_model.inputs,
         #outputs=initial_model.get_layer(name="my_intermediate_layer").output,
         #)
         # Call feature extractor on test input.
         # x = tf.ones((1, 250, 250, 3))
         # features = feature_extractor(x)outLabel = out(inputs)
         # array of output of all layers.
         outLabel = []
         weights  = [] #array of Tuple/list of tuple
         for layerIndx , layer in enumerate(layers):
         
             if layerIndx == 0 :
                 continue

             #if isinstance(layer,self.model) :
             #    set_trainable(layer)

             print("layer index", layerIndx, "layer name", layer.name)
             # weight   inputxoutput
             if str(layer.name)[0:4] == 'conv':
                W    = layers[layerIndx].get_weights()[0]
                B    = layers[layerIndx].get_weights()[1]
                temp = list(zip(W,B))
                weights.append(temp)
                """
                print(weights)
                fig = plt.figure()
                ax = plt.axes()

                #in1 = tf.Variable([16])
                #in2 = tf.Variable([16])
                #ax.plot3D(W[0],'green')
                for index1 , xy in enumerate(W):
                     for index2, x in  enumerate(xy):
                         for y in x:
                           print(xy.shape,x.shape,y.shape)
                           ax.plot(W[index1][0][0],W[0][index2][0],y,'green')

                plt.show()
                """
             with tf.GradientTape() as tape:
                  out    = self.mmodel.get_layer(layer.name).output 
                  #weight = self.model.get_layer(layer.name).get_weights()[0]
                  #bias   = self.model.get_layer(layer.name).get_weights()[1]
            
                  #inputss= self.model.get_layer(layer.name).input
            
                  #outputs=self.model.get_layer(layer.name).output
                  #Get the currnet layer..funcation and call this function.
                  # for given input, what is currnet layer output.. so input will be always fixed,
                  #for given input any layer will have difftent out... so every time
                  # we given same input and but look any currnet layer output
                  layerM = Model(inputs=self.mmodel.input, outputs=out)
                  #y_pred = model.predict(traininput, training=True)  # Forward pass
                  if isinstance(layerM, Model) :
                      #set_trainable(layerM)
                      layerM.trainable = True


                  #33predictedOut = layerM(PrvActivation)
                  #predictedOut    = layerM.predict(inputs, training=True)
                  predictedOut    = layerM.predict(inputs)
                  #Yu can plot only filter weigts such conv2d. or else

         print("outlabe shape", outLabel)
         #last layer
         return [outLabel, predictedOut, weights]

    def evaluate_model(Self):
        if not os.path.exist(DATA_PATH):
            raise FileNotFoundError(f"Evaluation data not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)
        x_data = np.random.rand(len(df),28,28,3).astype(np.float32)
        y_data = pd.get_dummies(df['label']).values

        _, self.x_test, _, self.y_test= train_test_split(x_data,y_data,test_size=0.2,random_state=42)
        print("starting model evaluation..")
        loss, accuracy = self.mmodel.evaluate(self.x_test,self.y_test,verbose=1)
        return {"loss":loss,"accuracy":accuracy}


    def predict_image(self, image_byte):

        # It should be a NumPy array or TensorFlow tensor.

        # Resize the image to 32x32 pixels
        # The `method` parameter controls the interpolation algorithm
        image_32x32 = tf.image.resize(image_bytes, (32, 32))

        # Add a batch dimension to the image (e.g., from (32, 32, 3) to (1, 32, 3, 3))
        image_for_prediction = tf.expand_dims(image_32x32, axis=0)

        # Make a prediction with the resized image
        predictions = self.mmodel.predict(image_for_prediction)

        #image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        #image = image.resize((32,32))
        #input_array = np.array(image,dtype=np.float32)/255.0 
        #input_array = np.expand_dims(input_array,axis=0)
        #predictions = self.mmodel.predict(input_array)
        predicted_class = np.argmax(predictions)
        return int(predicted_class)



    """    
    def computeBackpropgation(loss):
        
        #compute divergence of loss , say l2 DIVERGENCE or l1  

        for every layer:
         
            #comute gradetn of loss wrt weiggst of this layer
            layer.gradient=tapegradient(loss)

            #update leraning rate
            #update weight of this layer

            #by this time all llayer weigts are updateed incliding bias weitgs



         


    def trainModel(input, output):

        x_predict = computeforwardath(input, model)

        #compute loss = input - x_predict

        # update weiths of each layer
        #compute backpropgation

    """

    #comute forwarad path
    """    
    def prediction():
        #tbd

    def etStatus(typeofstatus):
        #tbd


    # HTTP REQ from clinet
    def trainRequest():
        #tbd


    # htpp request
    def pridictInferenceReq():
        #tbd

    """
    def MeltsImageTrainProcess(self):
 
        #Create models , this is uncompiled model
        #mmodel = self.resnet_v1(self.input_shape,int(self.depth))
        #Read data base and preproces and sepatare as train and test data
        #getDataset(self)
        testV = keras.datasets.cifar10
        (self.x_train, self.y_train),(self.x_test,self.y_test) = testV.load_data()

        #get as tuple of  data set as training and test
        self.datasetTrain=tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))

        #shufflle and slices.
        self.datasetTrain=self.datasetTrain.shuffle(buffer_size=1024).batch(64)


        #get as tuple of test
        self.datasetTest=tf.data.Dataset.from_tensor_slices((self.x_train,self.y_train))
        self.datasetTest=self.datasetTest.shuffle(buffer_size=1024).batch(64)

        #model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)
        self.mmodel.compile( optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                       loss=keras.losses.MeanSquaredError(),
                       metrics=[keras.metrics.SparseCategoricalAccuracy()],

                       )

        print("shape x_train", self.y_train.shape)
        #breakpoint()
        self.history=self.mmodel.fit(self.datasetTrain, epochs=1)
        # making epoch 1 , to sped up frame work.
        print("test model or evulaute")
        results=self.mmodel.evaluate(self.datasetTest)

        os.makedirs("models", exist_ok=True)
        self.mmodel.save(MODEL_PATH)
        # Save the model to the standard SavedModel format
        # This creates a directory structure containing the model's graph and weights
        #self.mmodel.save('saved_model_melts', save_format='.keras')
        plot_model(self.mmodel, "ModelMelts.png")
        # Assume 'model' is your trained Keras model

        # Create a TFLiteConverter from the Keras model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.mmodel)

        # Convert the model
        tflite_model = converter.convert()

        # Save the .tflite file
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)

        print("Model converted and saved to 'model.tflite'.")

        return self.datasetTrain,self.datasetTest

        #train the model as backpropagaton 
        #self.meltsTraining(self.model, self.x_train,self.y_train)

        #Comile the model, with loss so that fit shouldnot be null
        #
        # optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        #loss=keras.losses.SparseCategoricalCrossentropy(),
        #metrics=[keras.metrics.SparseCategoricalAccuracy()],

        #model.compile( optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        #               loss=keras.losses.MeanSquaredError(),
        #               metrics=[keras.metrics.SparseCategoricalAccuracy()],
        #             )

        #model.summary()

        #print("traing which further need to be subcalssed")
        #does traing on trained data set
        #m3odel.fit(datasetTrain, epochs=3)

        #print("test model or evulaute")
        #results=model.evaluate(datasetTest)


        """
        for testTrain  in self.x_test:
            outLabel = self.ComputeforwardPath(testTrain)
            plt(outLabel)
         """

        #plot the out put






#main
stackN = 3
def main():

    input_shape = (32,32,3)
    depth = stackN * 6 + 2
    modelMelts = melts(input_shape, depth)

    # caaling call function 
    #output     = modelMelts(tf.keras.Input)


    # modelMelts.compile( optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    #                    loss=keras.losses.MeanSquaredError(),
    #                    metrics=[keras.metrics.SparseCategoricalAccuracy()],

    #                   )

    datasetTrain,datasetTest = modelMelts.MeltsImageTrainProcess()
    
    #os.makedirs("model", exist_ok=True)
    #model_path= "models/modelMelts.keras"

    #modelMelts.save(model_path)
    print(f"Custom model saved to {MODEL_PATH}")

    # Assume 'model' is your trained Keras model
    # For example:
    # model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    # model.compile(optimizer='sgd', loss='mean_squared_error')
    # model.fit([1, 2, 3, 4], [2, 4, 6, 8], epochs=1)
    # Save the model to the standard SavedModel format
    # This creates a directory structure, not a single file
    #model.save('saved_model_melts', save_format='tf')
    #print("Model saved directly to SavedModel format in 'saved_model_melts' directory.")
    #import tensorflow as tf
    #from tensorflow import keras

    # Assume 'model' is your trained Keras model

    # Create a TFLiteConverter from the Keras model
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Convert the model
    #tflite_model = converter.convert()

    # Save the .tflite file
    #with open('model.tflite', 'wb') as f:
    #    f.write(tflite_model)

    #print("Model converted and saved to 'model.tflite'.")


@router.post("/train", summary="Train the model with new data")
async def train_model_endpoint():
          
        try:
            input_shape = (32,32,3)
            depth = stackN * 6 + 2
            model = melts(input_shape, depth)
            model.MeltsImageTrainProcess()
            return {"message": "Training completed and model saved "}
        except FileNotFoundError as e:
            raise HTTPException(status_code=404,detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


#, @router.get("/evaluate", summary="Evaluate the current model") is a decorator used to define a specific API endpoint. It tells FastAPI how to handle an HTTP GET request made to the /evaluate path

@router.get("/evaluate", summary="Evaluate the currnet model")
async def evaluate_model_endpoint():
        try:
             input_shape = (32,32,3)
             depth = stackN * 6 + 2
             model = melts(input_shape, depth)
             result= model.evaluate()
             return {"message": "Evaluation completed ", **result}
        except FileNotFoundError as e:
             raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))




# --- API Endpoint ---
@router.post(
    "/predict",
    summary="Make a prediction from an uploaded image",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict_image_endpoint(file: UploadFile = File(...)):
    
    input_shape = (32,32,3)
    depth  = stackN * 6 + 2
    model  = melts(input_shape, depth)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not available."
        )
    
    # 1. Validate file content type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Only JPEG, PNG, and WEBP are supported."
        )

    try:

        # 2. Read file content asynchronously and process
        content = await file.read()
        
        # 3. Use the pre-loaded model to make a prediction
        prediction = model.predict_image(content)
        
        return {"prediction": prediction}
        
    except ValueError as ve:
        # Handle specific exceptions related to image processing
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        # Catch any other unexpected exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )





# Endpoint for retrieving the log file contents
@router.get("/get_log", summary="Retrieve application logs")
async def get_log(api_key: Optional[str] = Header(None)):
    """
    Retrieves the contents of the application log file.
    Access to this endpoint should be restricted for security.
    """
    # Simple API key authentication for demonstration purposes
    # In production, use a proper authentication method (e.g., OAuth2)
    SECRET_KEY = os.getenv("LOG_ACCESS_KEY", "your-secure-log-key")
    if api_key != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    log_file_path = "meltlogdata.log"
    if not os.path.exists(log_file_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    try:
        with open(log_file_path, "r") as f:
            logs = f.readlines()
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Failed to read log file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read log file: {e}")



@router.get("/get_plot", summary="Generate and return a sample plot")
async def get_plot():
    """Generates a sample plot using matplotlib and returns it as a PNG image."""
    try:

        input_shape = (32,32,3)
        depth  = stackN * 6 + 2
        model  = melts(input_shape, depth)
        buffer =  model.getHistory()
         # Save the plot to an in-memory buffer
        # Return the buffer content as a StreamingResponse
        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate plot: {e}")


app.include_router(router)

if __name__=='__main__':
      main()
    





         




