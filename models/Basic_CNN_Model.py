from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


class Basic_CNN_Model:
    """
    Basic model class initialises parameters and
    builds its model with Keras (see documentation on https://keras.io/)
    having 4 hidden layers created with convolutional, pooling and dropout sublayers
    and a global average pooling flatten layer.
    """
    
    def __init__(self, name=None, metric='accuracy'):
        """Initialize parameters and model"""
        self.model_name = name
        self.metric = metric
        self.opt = None
        self.optAdam = Adam()
        self.build_model()
        
    
    def build_model(self):        
        # layer foundation 
        self.model = Sequential()

        # input layer
        # several combinations of Convolutional and MaxPooling layers as hidden layers
        self.model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                 input_shape=(224, 224, 3)))
        self.model.add(MaxPooling2D(pool_size=2, strides= 2, padding='same'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides= 2, padding='same'))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides= 2, padding='same'))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides= 2, padding='same'))
        self.model.add(Dropout(0.4))
        self.model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides= 2, padding='same'))
        self.model.add(Dropout(0.5))

        self.model.add(GlobalAveragePooling2D())  # flatten layer 
        self.model.add(Dense(2, activation='softmax'))  # total connected layer with 2 total chest categories
        
        # print model information
        print("\n--- Build model summary of Basic_CNN_Model: ---")
        self.model.summary()        
      
    def get_model(self):
        return self.model
    
    
    def train_model(self, model, epochs, batch_size, filepath, train_tensors, train_targets, valid_tensors, valid_targets):
        # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True)
        history = model.fit(train_tensors, train_targets, 
                                 validation_data=(valid_tensors, valid_targets),
                                 epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=2)
        self.model = model
    
        return history
    
    
    def augmentation_train_model(self, model, filepath, training_data, validation_data, epochs, train_tensors, batch_size):
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True)
        
        # example: train on 5216 samples and batch_size=32 => 5216/32 = 163 as steps_per_epoch
        base_model_aug_history = model.fit_generator(training_data,
                                 validation_data=validation_data, validation_steps=0.5,
                                 epochs=epochs, steps_per_epoch=train_tensors.shape[0]//batch_size,
                                 callbacks=[checkpointer], verbose=2)
        self.model = model
        
        return base_model_aug_history
    
    
    def set_Adam_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False):
        # default: keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # to avoid overfitting:
        # let's train the model using Adam with decay 1e-6 for each learning update
        self.optAdam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
    

    def compile_model(self, model, optimizer, loss='binary_crossentropy', metrics=['accuracy']):
        if optimizer in ['Adam']:
            # Adam optimizer with loss='binary_crossentropy', metrics=['accuracy']
            model.compile(loss=loss, optimizer=self.optAdam, metrics=metrics)
        elif optimizer in ['rmsprop']:
            model.compile(loss=loss, optimizer='rmsprop', metrics=metrics)
        else:
            print("Improved_CNN_Model: unknown optimiser, compile not possible!")
            
        self.model = model
    
    
    def load_best_weights(self, model, filepath):
        model.load_weights(filepath)        
        self.model = model