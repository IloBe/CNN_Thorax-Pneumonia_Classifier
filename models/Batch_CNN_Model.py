from keras import layers, models
from keras.optimizers import Adam
from keras.models import Sequential

class Batch_CNN_Model:
    """
    Batch model class initialises parameters and
    builds models with Keras (see documentation on https://keras.io/)
    """
    
    def __init__(self, name=None, metric='accuracy'):
        """Initialize parameters and model"""
        self.model_name = name
        self.metric = metric
        self.build_model()
        self.compile_model()
        
    
    def build_model(self):        
        # layer foundation 
        self.model = Sequential()

        # input layer
        # and output arrays of shape (*, 16)
        self.model.add(Dense(16, input_dim = 4, activation = "relu")) 
        
        # hidden layers
        # BatchNormalization included to improve performance and stability, and it has to be included
        # after each fully-connected layer, but before the activation function and dropout
        # Dropout layers are added to avoid overfitting
        
        self.model.add(Dense(32))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.3))
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.4))
        self.model.add(Dense(128))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.6))
        
        # output layer with number of 2 categories
        self.model.add(Dense(2, activation='softmax'))
        
        # print model information
        print("\n--- Build model summary of Batch_CNN_Model: ---")
        self.model.summary()
        
     
    def compile_model(self):
        # initiate Adam optimizer
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
        # let's train the model using Adam with decay
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=[self.metric])
        
        
    def get_model(self):
        return self.model