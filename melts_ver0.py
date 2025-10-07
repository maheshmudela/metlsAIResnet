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
from tensorflow.keras.layers import Dropout, Dense,Conv2D, BatchNormalization, Activation
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

num_classes= 10
batch_size = 32
epochs     = 10

"""
W = 3 X 4 , input = (3,),  output = (,4)
class Dense(tf.Module):
  def __init__(self, in_features, out_features, name=None):
    super().__init__(name=name)
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)

"""

class testM(Model):

     def __init__(self, inputs_shape, depth):

         super().__init__()

         self.lossfn = tf.keras.losses.CategoricalCrossentropy(
                        from_logits=False, label_smoothing=0.0, axis=-1
                        )

         #self.lossfn  = tf.keras.losses.SparseCategoricalCrossentropy(
         #               from_logits=False,
         #               reduction=None,
         #               name="sparse_categorical_crossentropy"
         #               )

         #self.lossfn      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='sum_over_batch_size')
         self.losstracker = tf.keras.metrics.Mean(name='loss')
         self.train_acc_metric = tf.keras.metrics.Accuracy(name='accuracy')
         self.val_acc_metric = tf.keras.metrics.Accuracy(name='val_accuracy')
         
         # 16 , 32x32  out , 
         conv = Conv2D(filters=16,  # no of units .. so that con2d out is 1 , 32*32
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

         inputs = tf.keras.Input(shape=inputs_shape)

         
         x = conv(inputs)

         print("shape of con2d out", x.shape)

         """
         32x32x3

         

                 [3x3] ....[3x3]

                 ..

                 [3x3]    
         """

         x = BatchNormalization()(x)

         print("shape of con2d out", x.shape)

         #denseH = Dense(10, activation="relu")

         #feature = denseH(x)
         #print("shape ofdenseH out", feature.shape)

         out = Dropout(0.5) (x)
         #print("shape of drop out", out.shape)

         out = AveragePooling2D(pool_size=2)(out) # downsample /2
         print("shape of pool out", out.shape)

         out     = Flatten() (out)
         print("shape of flatten out", out.shape)

         #last dense layer
         denseO = Dense(num_classes, activation="softmax", kernel_initializer='he_normal')

         out = denseO(out)
         print("shape of dens0 out",out.shape)

         self.model = Model(inputs=inputs, outputs=out)
 
         #Create models , this is uncompiled model
         #mmodel = self.resnet_v1(self.input_shape,int(self.depth))
         #Read data base and preproces and sepatare as train and test data
         #getDataset(self)
 
     def call(self, inputs):

         shape = inputs.shape

         #x = tf.expand_dims(inputs, axis=-1)
         out = self.model(inputs)
         return out 

     @property
     def metrics(self):
         
         return [self.losstracker]

    
     def reset_metrics(self):
        for m in self.metrics:
            #m.reset_state()
            print("rseting")


     def train_step(self, traindataSet) :

          x_train, y_train = traindataSet

          #self.zero_grad()
          #x_train = tf.expand_dims(x_train, axis=0)
           
          with tf.GradientTape() as tape:
              y_pred = self(x_train, training=True)
              #print(" y_pred shape", y_pred.shape, "y pred val ", y_pred, "x_true shape ", x_train.shape, "y true val", x_train)
              totalloss = self.lossfn(y_train,y_pred)
              #print("loss", totalloss, "lossshape", totalloss.shape)
             
          #totalloss.backward()
          grads=tape.gradient(totalloss, self.trainable_weights)

          #for i , val in enumerate(grads):
          #    print("i", i, "grads", val)


          trainVariable = self.trainable_weights

          self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

          # Update training metric.
          self.train_acc_metric.update_state(y_train, y_pred)

          # Log every 200 batches.
          # Display metrics at the end of each epoch
          train_acc = self.train_acc_metric.result()
          print("Training acc over epoch:",train_acc)

          # Reset training metrics at the end of each epo
          self.train_acc_metric.reset_state()

          print("totalloss", totalloss)

          return {"loss": totalloss, "accuracy": train_acc,}
    

     def getHistory(self,history):

            # Fit the model
            #history = AiModel.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
            # list all data in history
            print(history.history.keys())
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            #plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            #plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()



def main():

        # none as batch??
        input_shape = (32,32,3)

        # 
        modelM = testM(input_shape, 3)
        
        modelM.compile(optimizer=Adam(0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"] )

        #modelM.compile(optimizer="adam",loss="mse", metrics=["mae"], run_eagerly=True )
        #modelM.compile(optimizer="sgd", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[["mean_absolute_error"], ["accuracy"]] )


        modelM.summary()
        testV = tf.keras.datasets.cifar10
        (x_train, y_train),(x_test,y_test) = testV.load_data()
        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)


        #get as tuple of  data set as training and test
        datasetTrain = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        #shufflle and slices.
        datasetTrain = datasetTrain.shuffle(buffer_size=1024).batch(batch_size)

        #get as tuple of test
        datasetTest  = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        datasetTest  = datasetTest.shuffle(buffer_size=1024).batch(batch_size)

        for i , (x,y) in enumerate(datasetTrain):
             print(" batch",i, "x shape", x.shape, "y shape", y.shape)



        #trainX, testX = modelM.testVector()

        #modelM.fit(trainX, epochs=1, batch_size =128)

        history=modelM.fit(datasetTrain, batch_size=batch_size, epochs=10, verbose=0, validation_data=(datasetTest),shuffle=True)

        for key, value in history.history.items():
            print("key", key, "value", value)


        modelM.getHistory(history)
        print("train done")

        #modeM.evaluate()

        #modeM.predict()




if __name__=='__main__':
      main()
 



          


            


        
         


