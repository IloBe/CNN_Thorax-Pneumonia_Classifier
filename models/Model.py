from models.Batch_CNN_Model import Batch_CNN_Model
from models.Basic_CNN_Model import Basic_CNN_Model
from models.Improved_CNN_Model import Improved_CNN_Model
from models.Transfer_ResNet_CNN_Model import Transfer_ResNet_CNN_Model
from models.Transfer_InceptionV3_CNN_Model import Transfer_InceptionV3_CNN_Model

class Model:
    """ Model factory class for deep learning network types. """
   
    def __init__(self, type_name=None, name=None, metric='accuracy'):
        '''
        Initialisation of the network model

        Params:
        - type_name: general network type
        - name: specific network that shall be created
        - metric: metric type depends on the task and the used optimiser,
          e.g. for classifiction the simplest one is 'accuracy'
        '''
        self.model_class = None
        self.model = None
        self.name = name
        self.metric = metric
        self.type_name = type_name
        
        if self.type_name in ["Basic"]:
            self.model_class = Basic_CNN_Model(name=name, metric=metric)
        elif type_name in ["Batch"]: 
            self.model_class = Batch_CNN_Model(name=name, metric=metric)
        elif type_name in ["Improved"]:
            self.model_class = Improved_CNN_Model(name=name, metric=metric)
        elif type_name in ["Transfer"]:
            if self.name in ["Transfer_ResNet_CNN_Model"]:
                self.model_class = Transfer_ResNet_CNN_Model(name=name, metric=metric)
            elif self.name in ["Transfer_InceptionV3_CNN_Model"]:
                self.model_class = Transfer_InceptionV3_CNN_Model(name=name, metric=metric)
        else:
            print("Wrong Model type - {} -, does not exist, therefore no model building possible.".format(type_name))

            
    def get_model(self):
        # Can return None, if model type does not exist yet.
        self.model = self.model_class.get_model()
        return self.model
    
    def get_class(self):
        # Can return None, if model type does not exist yet.
        return self.model_class
    
