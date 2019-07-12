from keras.applications.inception_v3 import InceptionV3
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

class Transfer_CNN_Model:
    """
    Transfer learning model class initialises parameters and
    builds its model with Keras (see documentation on 
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
    
    As pre-trained model InceptionV3 is included, using Adam as optimiser and training happened without Augmentation.
    """
    
    def __init__(self, name=None, metric='accuracy', train_inceptV3=None):
        """Initialize parameters and model"""
        self.model_name = name
        self.metric = metric
        self.train_inceptV3=train_inceptV3
        self.optAdam = Adam()
        self.build_model()
        
    
    def build_model(self):
        # layer foundation 
        self.model = Sequential()

        # transfer and hidden layers
        self.model.add(GlobalAveragePooling2D(input_shape=self.train_inceptV3.shape[1:]))
        self.model.add(BatchNormalization())
        self.model.add(Dense(2, activation='softmax'))  # total connected layer with 2 total chest categories
        
        # print model information
        print("\n--- Build model summary of Transfer_CNN_Model: ---")
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
            print("Improved_CNN_Model: unknown optimiser, compile not possible!")
            
        self.model = model
        
        
    def train_model(self, model, epochs, batch_size, filepath, train_tensors, train_targets, valid_tensors, valid_targets):
        # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True)
        transfer_model_history = model.fit(train_tensors, train_targets, 
                                 validation_data=(valid_tensors, valid_targets),
                                 epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=2)
        self.model = model
    
        return transfer_model_history
        
        
    def load_best_weights(self, model, filepath):
        model.load_weights(filepath)        
        self.model = model