from keras.layers import Dropout, Dense, BatchNormalization, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

class Batch_CNN_Model:
    """
    Batch model class initialises parameters and
    builds its model with Keras (see documentation on https://keras.io/).
    
    As described in https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    such architecture shall create the bottleneck features. This concept shall prevent us to use augmentation.
    
    In the hidden layers, BatchNormalization instances are included to improve performance and stability.
    Don't reduce batch size smaller 32, because then performance is decreasing (according literature and blogs about
    BatchNormalization, see
    https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)
    """
    
    def __init__(self, name=None, metric='accuracy'):
        """Initialize parameters and model"""
        self.model_name = name
        self.metric = metric
        self.optAdam = Adam()
        self.build_model()
        
    
    def build_model(self):        
        # layer foundation 
        self.model = Sequential()

        # input layer
        self.model.add(Conv2D(filters=32, kernel_size=2, padding='same',
                              activation='relu', 
                              input_shape=(224, 224, 3)))
        self.model.add(Conv2D(filters=32, kernel_size=2,
                              activation='relu', 
                              input_shape=(224, 224, 3)))
        
        # hidden layers
        # BatchNormalization included to improve performance and stability, 
        # it normalizes the output of a previous activation layer by subtracting the batch mean and dividing
        # by the batch standard deviation, being the input for the next activation function.
        # It has to be included after each fully-connected layer, but before the activation function and dropout.
        # Dropout layers are added to avoid overfitting.
        # According literature, batch size don't have to be small (don't reduce the value 32), because then
        # BatchNormalization reduces the performance;
        # default params are used: Conv2D - channels last; BatchNormalization - axis -1
                
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))       
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.4))
        self.model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
        self.model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=2, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.6))
        self.model.add(Conv2D(filters=512, kernel_size=2, padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=2, activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=2, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu")) 
        self.model.add(Dropout(0.7))
        
        self.model.add(GlobalAveragePooling2D())  # flatten layer 
        self.model.add(Dense(2, activation='softmax'))  # total connected layer with 2 total chest categories
        
        # print model information
        print("\n--- Build model summary of Batch_Model: ---")
        self.model.summary()
        
    
    def get_model(self):
        return self.model

    
    def set_Adam_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False):
        # default: keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # to avoid overfitting:
        # let's train the model using Adam with decay 1e-6 for each learning update
        self.optAdam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
     
    
    def compile_model(self, model, optimizer, loss='binary_crossentropy', metrics=['accuracy']):
        if optimizer in ['Adam']:
            # for learning: Adam optimizer with loss='binary_crossentropy', metrics=['accuracy']
            model.compile(loss=loss, optimizer=self.optAdam, metrics=metrics)
        else:
            print("Batch_CNN_Model: unknown optimiser, compile not possible!")
            
        self.model = model
    
        
        
    def train_model(self, model, filepath, train_tensors, train_targets, valid_tensors, valid_targets, epochs, batch_size):
        # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True)
        batch_model_history = model.fit(train_tensors, train_targets, 
                                        validation_data=(valid_tensors, valid_targets),
                                        epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=2)
        self.model = model
        
        return batch_model_history
    
        
    def load_best_weights(self, model, filepath):
        model.load_weights(filepath)        
        self.model = model
    