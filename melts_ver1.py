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


class  melts:

    def __init__(self,inputshape,depth):
        self.input_shape = inputshape
        self.depth       = depth
        print("constructor or init")


    def getHistory(self,history):

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

        for layerIndex , layerName in enumerate(self.model.layer):

            # every layer have 2d weights
            self.weight_dict["weight" + str(layerName)]  = self.model.layer[layerIndex].get_weight()[0]
            self.weight_dict["bias" + str(layerName)]    = self.model.layer[layerIndex].get_weight()[1]
      
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




    def meltsTraining(self, model, xTrain,yTrainLabel):
         # Unpack the data. Its structure depends on your model and
         # on what you pass to `fit()`.

             
         # for every traing sample
         for index, train in enumerate(xTrain):  #for every train signal  

            label = yTrainLabel[index]
            ytrue = tf.reshape(label,shape = (1,-1)) 
            trainSample = tf.reshape(train,shape=(1,32,32,3))

            # forward path..computataion, this shoulld predict signle vaalue , based last layer as softamx? or 
            # sigmoid.. but seems it is predicting 10 value instead of 1 , so either use 
            # your own forward path or fix this..??
            #y_pred = model.predict(trainSample)  # Forward pass
            #y_out  = tf.keras.activations.softmax(y_pred, axis=-1)

            allLayer, y_out, weights =self.ComputeforwardPath(trainSample)
            """
            for index, layer in enumerate(allLayer):
                print("layer", layer, "shpae", layer.shape)
                #plot.subplot(allLayer[index])
                self.plot_layer_outputs(layer)
                breakpoint()
                self.plot_imshow(layer)
                breakpoint()
                      
            """
            weightF = weights[index]
            #plot_weights(weightF)

            #if index > 1 and index < 50 :
            print("outshape", y_out.shape, "input shape", ytrue.shape)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            #compute back path
            #trainable_vars = model.trainable_variables
            #print(trainable_vars)

            w = tf.Variable(tf.random.normal((10,10),dtype=tf.float32))

            with tf.GradientTape() as tape:
                y_predict = tf.nn.softmax(y_out)
                print("softmax", y_predict)
                #diff = tf.Variable(tf.random.normal((1,10), dtype=tf.int8))
                yt      = tf.Variable([[ytrue,ytrue,ytrue,ytrue,ytrue,ytrue,ytrue,ytrue,ytrue,ytrue]], dtype=tf.float32)
                yExpect =  tf.reshape(yt,shape=(1,10))

                #yExpect = [ytrue for i in range(y_predict.shape[1])]
                
                diff = tf.Variable(tf.random.normal((1,10), dtype=tf.float32))
                #print("yExpect shape", yExpect, y_predict)
                #cce = tf.keras.losses.CategoricalCrossentropy()
                #diff = cce(y_predict, yExpect)
                diff = y_predict # tf.dtypes.saturate_cast(tf.reduce_mean(tf.math.square(yExpect - y_predict)), tf.float32)
                #diff = tf.dtypes.saturate_cast(tf.reduce_mean(tf.math.square(yExpect - y_predict)), tf.float32)

                print("loss", diff)

                #out = tf.keras.ops.argmax(y_out,axis=0)
                #ypred = tf.reshape(out,shape=(1,1))

                #print("out ", ypred, "input ", ytrue)

                #loss = abs(tf.cast(yExpect,tf.int32) - tf.cast(ypred,tf.int32));
                #loss   = lossfn(labeloutput, y_pred)
                trainable_vars = model.trainable_variables

                tWeights = [tf.convert_to_tensor(trainable_vars[i]) for i in range(16)]
               

                print(tWeights)
  
                breakpoint()
                tape.watch(tWeights)

                # Compute gradients
                # how many train varibale in model, of 16 layers:
                # compute graduant of loss wrt weigts  
                gradients = tape.gradient(diff,tWeights)
   
                # Update weights
                #optimizer = tf.keras.optimizers.SGD()
                print("gradients ", gradients)
                optimizer = keras.optimizers.Adam(learning_rate=1e-3)

                breakpoint()
                optimizer.apply(gradients,trainable_vars)
            # Update metrics (includes the metric that tracks the loss)
            for metric in self.metrics:
                if metric.name == "loss":
                    metric.update_state(loss)
                else:
                    metric.update_state(y, y_pred)
            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}


    def resnet_v1(self,input_shape, depth, num_classes=10):
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

        if((depth - 2) % 6) != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

        # Start model definition.
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)
        inputs = tf.keras.Input(self.input_shape)

        # inputs = Input(shape=input_shape)
        #inputs = tf.reshape(inputs, [0, 32, 32, 3])

        x = self.resnet_layer(inputs,num_filters)

        print("layer1", x.shape);

        # Instantiate the stack of residual units
        for stack in range(3):
            print(" no of stack ", stack, num_res_blocks)
            for res_block in range(num_res_blocks):

                print(" num_res_blocks", num_res_blocks)
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

        return model


    def GetModel(self):
        return self.model
        
        


    def getDataset(self): 

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

         layers = self.model.layers
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

             out    = self.model.get_layer(layer.name).output 
             #weight = self.model.get_layer(layer.name).get_weights()[0]
             #bias   = self.model.get_layer(layer.name).get_weights()[1]
            
             #inputss= self.model.get_layer(layer.name).input
            
             #outputs=self.model.get_layer(layer.name).output
             #Get the currnet layer..funcation and call this function.
             # for given input, what is currnet layer output.. so input will be always fixed,
             #for given input any layer will have difftent out... so every time
             # we given same input and but look any currnet layer output
             layerM = Model(inputs=self.model.input, outputs=out)
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

    def getStatus(typeofstatus):
        #tbd


    # HTTP REQ from clinet
    def trainRequest():
        #tbd


    # htpp request
    def pridictInferenceReq():
        #tbd

    """
    def MeltsImageTrainProcess(self):

        #Read data base and preproces and sepatare as train and test data
        #getDataset(self)
        testV = keras.datasets.cifar10
        (self.x_train, self.y_train),(self.x_test,self.y_test) = testV.load_data()
     
        print("shape x_train", self.y_train.shape)
        #breakpoint()
      
        #Create models
        self.model=self.resnet_v1(self.input_shape,int(self.depth))
        
        #train the model as backpropagaton 
        self.meltsTraining(self.model, self.x_train,self.y_train)
 
        #test model
        for testTrain  in self.x_test:
            outLabel = self.ComputeforwardPath(testTrain)
            plt(outLabel)


        #plot the out put






#main
n = 3
def main():

    input_shape = (32,32,3)
    depth = n * 6 + 2
    modelMelts = melts(input_shape, depth)
    modelMelts.MeltsImageTrainProcess()
 





if __name__=='__main__':
      main()
    





         




