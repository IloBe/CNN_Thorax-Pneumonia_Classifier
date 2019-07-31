from sklearn import metrics
from models.Model import Model

import seaborn as sns
sns.set_style('darkgrid')

import matplotlib.pyplot as plt                        
import numpy as np



class Evaluation_Class:
    '''
    Delivers a set of metric calculations.
    '''
    
    
    def print_test_accuracy(model_name, predictions, test_targets):
        # report test accuracy
        test_accuracy = 100 * np.sum(np.array(predictions) == np.argmax(test_targets, axis=1))/ \
                                                                        len(predictions)
        print('{}: Test Accuracy: {:3.2f}'.format(model_name, test_accuracy))
    
    
    def show_history_accuracy(history):
        # show the history diagram for epochs and associated accuracy results (coding of the Keras documentation)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.show()
        
        
    def show_history_loss(history):
        # Show the training & validation loss values (learning)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
        

    def get_other_metrics(test_targets, predictions):
        ''' 
        Delivers textual prints of several metrics:
        precision, recall, f1 score, f-beta score with beta 0.5, 
        Cohens Kappa and Confusion Matrix of the test data
        '''

        # precision: tp / (tp + fp)
        precision = metrics.precision_score(test_targets[:,1], predictions)
        print('Precision: %.3f' % precision)
        # recall: tp / (tp + fn)
        recall = metrics.recall_score(test_targets[:,1], predictions)
        print('Recall: %.3f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = metrics.f1_score(test_targets[:,1], predictions)
        print('F1 score: %.3f' % f1)
        # f-beta score
        f_beta = metrics.fbeta_score(test_targets[:,1], predictions, average = 'weighted', beta=0.5)
        print('F-beta score: %.3f  with beta=0.5' % f_beta)

        print()
        # kappa
        kappa = metrics.cohen_kappa_score(test_targets[:,1], predictions)
        print('Cohens kappa: %.3f' % kappa)
        # confusion matrix
        matrix = metrics.confusion_matrix(test_targets[:,1], predictions)
        print('Confusion matrix of the test data:\n{}'.format(matrix))

    
    def get_classification_report(y_test, y_pred):
        print('\n\nClassification report (0 is NORMAL and 1 is PNEUMONIA)')
        print(metrics.classification_report(y_test, y_pred))
        
        
    def plot_ROC_AUC(fpr, tpr, modelname):
        print("--- associated ROC AUC diagram of model type {} ---".format(modelname))
        plt.plot(fpr, tpr, color='red')
        plt.title('Receiver Operating Characteristic Curve', size=15)
        plt.plot([0, 1], [0, 1], color='green', linestyle=':')
        plt.xlabel('False Positive Rate', size=13)
        plt.ylabel('True Positive Rate', size=13)
        plt.show()
        

        
    
    
