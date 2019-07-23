from keras.applications.resnet50 import ResNet50
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, InputLayer
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

class Transfer_ResNet_CNN_Model:
    """
    Transfer learning model class initialises parameters and
    builds its model with Keras (see documentation on 
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
    
    As pre-trained model ResNet50 is included, using Adam as optimiser and training happened without and
    with Augmentation.
    The bottleneck features are created building a model part as feature extractor for the non-augmentation part.
    """
    
    def __init__(self, name=None, metric='accuracy'):
        """ Initialize parameters and model """
        self.model_class_name = name
        self.metric = metric
        self.resNet_model_withoutTop = None  # for bottleneck feature extraction
        self.resNet_finetuned_model_withoutTop = None   # for fine tuned model
        self.optAdam = Adam()
        
    
    def build_model(self, resNet_model):
        ''' specific final top model part using the created bottleneck features of the resNet model '''
        input_shape = resNet_model.output_shape[1]
        
        # define top model
        final_resNet_model = Sequential()
        final_resNet_model.add(InputLayer(input_shape=(input_shape,)))
        final_resNet_model.add(BatchNormalization())
        final_resNet_model.add(Dense(2, activation='softmax'))  # total connected layer with 2 total chest categories
        
        self.model = final_resNet_model
        
        # print model information
        print("\n--- Build model summary of final RestNet Transfer_CNN_Model with top layer: ---")
        self.model.summary()
        
        
    def build_model_feature_extractor(self, input_shape):
        '''
        model necessary to build the bottleneck features and
        model used for augmentation
        '''
        resNet = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        output = resNet.layers[-1].output  # resNet.output
        output = GlobalAveragePooling2D()(output)
        self.resNet_model_withoutTop = Model(resNet.input, output)
        
        self.resNet_model_withoutTop.trainable = False
        for layer in self.resNet_model_withoutTop.layers:
            layer.trainable = False
               
        self.model = self.resNet_model_withoutTop
        # print model information
        print("\n--- Build model summary of RestNet Transfer_CNN_Model as feature extractor: ---")
        self.model.summary()
        
        
    def build_ResNet_aug_model(self, resNet_model):
        ''' specific final ResNet model not using bottleneck features, needed for augmentation '''
        input_shape = resNet_model.output_shape[1]
        final_resNet_model = Sequential()
        final_resNet_model.add(resNet_model)
        final_resNet_model.add(InputLayer(input_shape=(input_shape,)))
        final_resNet_model.add(BatchNormalization())
        final_resNet_model.add(Dense(2, activation='softmax'))  # total connected layer with 2 chest categories
        
        self.model = final_resNet_model
        # print model information
        print("\n--- Build model summary of augmented RestNet Transfer_CNN_Model: ---")
        self.model.summary()
                
        
    def build_finetuned_model(self, input_shape):
        resNet = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        output = resNet.layers[-1].output  # resNet.output
        output = GlobalAveragePooling2D()(output)
        self.resNet_finetuned_model_withoutTop = Model(resNet.input, output)
        
        self.resNet_finetuned_model_withoutTop.trainable = True
        set_trainable = False
        for layer in self.resNet_finetuned_model_withoutTop.layers:
            if layer.name in ['res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c']:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        
        self.model = self.resNet_finetuned_model_withoutTop
        # print model information
        print("\n--- Build model summary of fine-tuned RestNet Transfer_CNN_Model: ---")
        self.model.summary()
        
        return self.resNet_finetuned_model_withoutTop
        
        
    def get_model(self):
        return self.model
    
    def get_ResNet_featuremodel_withoutTop(self):
        return self.resNet_model_withoutTop
        
    
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
            print("Transfer_CNN_Model: unknown optimiser, compile not possible!")
            
        self.model = model
        
        
    def train_model(self, model, epochs, batch_size, filepath, train_tensors, train_targets, valid_tensors,
                    valid_targets):
        # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True)
        transfer_model_history = model.fit(train_tensors, train_targets, 
                                 validation_data=(valid_tensors, valid_targets),
                                 epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=2)
        self.model = model
    
        return transfer_model_history
    
    
    def augmentation_train_model(self, model, filepath, training_data, validation_data, epochs, batch_size,
                                 train_tensors, valid_tensors):
        # with original dataset: validation_steps = 0.5  (16 / 32 = 0.5)
        # but with modified dataset it is 1 (32/32) => we use: valid_tensors.shape[0]//batch_size
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True)       
        transfer_model_aug_history = model.fit_generator(generator=training_data,
                                                         steps_per_epoch=train_tensors.shape[0]//batch_size,
                                                         epochs=epochs, verbose=2,
                                                         callbacks=[checkpointer],
                                                         validation_data=validation_data,
                                                         validation_steps=valid_tensors.shape[0]//batch_size)
        self.model = model
        
        return transfer_model_aug_history
        
        
    def load_best_weights(self, model, filepath):
        model.load_weights(filepath)        
        self.model = model