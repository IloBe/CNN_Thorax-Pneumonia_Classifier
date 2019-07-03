from models.Batch_CNN_Model import Batch_CNN_Model

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
        self.model = None
        self.name = name
        self.metric = metric
        self.type_name = type_name
        
        if type_name in ["Basic"]:
            self.model = network1(name=name, metric=metric)
        elif type_name in ["Batch"]: 
            self.model = Batch_CNN_Model(name=name, metric=metric)
        else:
            print("Wrong Model type - {} -, does not exist, therefore no model building possible.".format(type_name))

            
    def get_model(self):
        # Can return None, if model type does not exist yet.
        return self.model
    
           