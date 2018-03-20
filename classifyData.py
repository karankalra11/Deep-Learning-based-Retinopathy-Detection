
# coding: utf-8

# Let's first load the training data, after processing it into a single file by executing processImages.py .
# We use load() for loading the processed file:

# In[ ]:


import os
import csv
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
#FTRAIN = "/home/clab/Downloads/train5_images_128_128_all_new.csv"
FTEST = "/home/cse/Downloads/output/test_images_128_128_all_new.csv"
"""Loads data from FTEST if *test* is True, otherwise from FTRAIN.
"""
#os.chdir('../')
def load(test=False):
    fname = FTEST 
    with open(fname, 'r') as f: 
        y=[]
        X=[]
        for row in csv.reader(f): #each row in the input file is added to X and y lists
            r = np.array(map(np.float32, row))
            y_ = r[0] #the first column is the label
            X_ = np.array(r[range(1,r.shape[0])]).reshape(1,128,128) #the rest of the columns are images
            X.append(X_)
            y.append(y_)
        #converting X and y to np.arrays
        y=np.array(y,dtype=np.uint8)
        X=np.array(X) 
        if test: 
            y=None #in test files label is irrelevant
        else: X, y = shuffle(X, y, random_state=42) #shuffle train records 
    return X, y


# After defining the load() method, we actually use it:

# In[ ]:


import sys
sys.setrecursionlimit(20000) # since we might go over the stack size limit
X, y = load()   


# In order to use on_epoch_finished and on_epoch_finished:
# 1. We will change some parameters(update_learning_rate and update_momentum) to become Theano shared variables so that they could be update on the fly:

# In[ ]:


import sys
import theano
def float32(k):
    return np.cast['float32'](k)


# 2. Passing a parametrizable class with a __call__ method as our callback. (Its parameters: nn, which is the NeuralNet instance itself, and train_history, which is the same as nn.train_history_).

# In[ ]:


class AdjustVariable(object):
    def __init__(self, name, start=0.1, stop=0.0001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)    
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


# We can now train aneuralnetwork model based on the train set. We start by defining the first network (net1) as follows:

# In[ ]:


from lasagne import layers
import lasagne.nonlinearities
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
	('conv4', layers.Conv2DLayer),
        ('pool4', layers.MaxPool2DLayer),
        ('dropout4', layers.DropoutLayer),
        ('conv5', layers.Conv2DLayer),
        ('pool5', layers.MaxPool2DLayer),
        ('dropout5', layers.DropoutLayer),        
	('hidden6', layers.DenseLayer),
        ('dropout6', layers.DropoutLayer),
        ('hidden7', layers.DenseLayer),
	('hidden8', layers.DenseLayer),
	('hidden9', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 128, 128),
    conv1_num_filters=32, conv1_filter_size=(3, 3),conv1_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool1_pool_size=(2, 2),
    dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2),conv2_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2),conv3_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    conv4_num_filters=256, conv4_filter_size=(2, 2),conv4_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool4_pool_size=(2, 2),
    dropout4_p=0.4,
    conv5_num_filters=512, conv5_filter_size=(2, 2),conv5_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool5_pool_size=(2, 2),
    dropout5_p=0.5,
    hidden6_num_units=500,
    dropout6_p=0.6,
    hidden7_num_units=1000,
    hidden8_num_units=1000,
    hidden9_num_units=1000,
    output_num_units=5, output_nonlinearity=lasagne.nonlinearities.softmax,
    update_learning_rate=theano.shared(float32(0.1)),
    update_momentum=theano.shared(float32(0.8)),
    regression=False,
    batch_iterator_train=BatchIterator(batch_size=300),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.1, stop=0.0001),
        AdjustVariable('update_momentum', start=0.8, stop=0.999),
        ],
    max_epochs=500,
    verbose=1,
    )


net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
	('conv4', layers.Conv2DLayer),
        ('pool4', layers.MaxPool2DLayer),
        ('dropout4', layers.DropoutLayer),
        ('conv5', layers.Conv2DLayer),
        ('pool5', layers.MaxPool2DLayer),
        ('dropout5', layers.DropoutLayer),        
	('hidden6', layers.DenseLayer),
        ('dropout6', layers.DropoutLayer),
        ('hidden7', layers.DenseLayer),
	('hidden8', layers.DenseLayer),
	('hidden9', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 128, 128),
    conv1_num_filters=32, conv1_filter_size=(3, 3),conv1_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool1_pool_size=(2, 2),
    dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2),conv2_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2),conv3_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    conv4_num_filters=256, conv4_filter_size=(2, 2),conv4_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool4_pool_size=(2, 2),
    dropout4_p=0.4,
    conv5_num_filters=512, conv5_filter_size=(2, 2),conv5_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool5_pool_size=(2, 2),
    dropout5_p=0.5,
    hidden6_num_units=500,
    dropout6_p=0.6,
    hidden7_num_units=1000,
    hidden8_num_units=1000,
    hidden9_num_units=1000,
    output_num_units=5, output_nonlinearity=lasagne.nonlinearities.softmax,
    update_learning_rate=theano.shared(float32(0.1)),
    update_momentum=theano.shared(float32(0.8)),
    regression=False,
    batch_iterator_train=BatchIterator(batch_size=300),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.1, stop=0.0001),
        AdjustVariable('update_momentum', start=0.8, stop=0.999),
        ],
    max_epochs=500,
    verbose=1,
    )

net3 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
	('conv4', layers.Conv2DLayer),
        ('pool4', layers.MaxPool2DLayer),
        ('dropout4', layers.DropoutLayer),
        ('conv5', layers.Conv2DLayer),
        ('pool5', layers.MaxPool2DLayer),
        ('dropout5', layers.DropoutLayer),        
	('hidden6', layers.DenseLayer),
        ('dropout6', layers.DropoutLayer),
        ('hidden7', layers.DenseLayer),
	('hidden8', layers.DenseLayer),
	('hidden9', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 128, 128),
    conv1_num_filters=32, conv1_filter_size=(3, 3),conv1_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool1_pool_size=(2, 2),
    dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2),conv2_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2),conv3_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    conv4_num_filters=256, conv4_filter_size=(2, 2),conv4_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool4_pool_size=(2, 2),
    dropout4_p=0.4,
    conv5_num_filters=512, conv5_filter_size=(2, 2),conv5_nonlinearity=lasagne.nonlinearities.leaky_rectify, pool5_pool_size=(2, 2),
    dropout5_p=0.5,
    hidden6_num_units=500,
    dropout6_p=0.6,
    hidden7_num_units=1000,
    hidden8_num_units=1000,
    hidden9_num_units=1000,
    output_num_units=5, output_nonlinearity=lasagne.nonlinearities.softmax,
    update_learning_rate=theano.shared(float32(0.1)),
    update_momentum=theano.shared(float32(0.8)),
    regression=False,
    batch_iterator_train=BatchIterator(batch_size=300),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.1, stop=0.0001),
        AdjustVariable('update_momentum', start=0.8, stop=0.999),
        ],
    max_epochs=500,
    verbose=1,
    )

net4 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
	('conv4', layers.Conv2DLayer),
        ('pool4', layers.MaxPool2DLayer),
        ('dropout4', layers.DropoutLayer),
        ('conv5', layers.Conv2DLayer),
        ('pool5', layers.MaxPool2DLayer),
        ('dropout5', layers.DropoutLayer),        
	('hidden6', layers.DenseLayer),
        ('dropout6', layers.DropoutLayer),
        ('hidden7', layers.DenseLayer),
	('hidden8', layers.DenseLayer),
	('hidden9', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 128, 128),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    conv4_num_filters=256, conv4_filter_size=(2, 2), pool4_pool_size=(2, 2),
    dropout4_p=0.4,
    conv5_num_filters=512, conv5_filter_size=(2, 2), pool5_pool_size=(2, 2),
    dropout5_p=0.5,
    hidden6_num_units=500,
    dropout6_p=0.6,
    hidden7_num_units=1000,
    hidden8_num_units=1000,
    hidden9_num_units=1000,
    output_num_units=5, output_nonlinearity=lasagne.nonlinearities.softmax,
    update_learning_rate=theano.shared(float32(0.1)),
    update_momentum=theano.shared(float32(0.8)),
    regression=False,
    batch_iterator_train=BatchIterator(batch_size=300),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.1, stop=0.0001),
        AdjustVariable('update_momentum', start=0.8, stop=0.999),
        ],
    max_epochs=500,
    verbose=1,
    )



# Let's train net1 according to the train data that we loaded:

# In[ ]:


#net1.fit(X, y)
import cPickle as pickle
with open('net011.pickle', 'rb') as f:
    net1 = pickle.load(f)
    
with open('net012.pickle', 'rb') as f1:
    net2 = pickle.load(f1)
    
with open('net1.pickle', 'rb') as f2:
    net3 = pickle.load(f2)

with open('net1my.pickle', 'rb') as f3:
    net4 = pickle.load(f3)    
#with open('net4.pickle', 'rb') as f3:
   # net4 = pickle.load(f3)
    
#with open('net5.pickle', 'rb') as f4:
    #net5 = pickle.load(f4)
#net1=pickle.load(net5.pickle)
# The output (50 first iterations) looks something like this:
#   input                 (None, 1, 128, 128)     produces   16384 outputs
#   conv1                 (None, 32, 126, 126)    produces  508032 outputs
#   pool1                 (None, 32, 63, 63)      produces  127008 outputs
#   conv2                 (None, 64, 62, 62)      produces  246016 outputs
#   pool2                 (None, 64, 31, 31)      produces   61504 outputs
#   conv3                 (None, 128, 30, 30)     produces  115200 outputs
#   pool3                 (None, 128, 15, 15)     produces   28800 outputs
#   hidden4               (None, 500)             produces     500 outputs
#   hidden5               (None, 500)             produces     500 outputs
#   output                (None, 5)               produces       5 outputs
#   epoch    train loss    valid loss    train/val    valid acc  dur
# -------  ------------  ------------  -----------  -----------  ------
#       1       0.88311       0.86840      1.01694      0.73472  69.54s
#       2       0.86898       0.86781      1.00135      0.73472  69.52s
#       3       0.86820       0.86693      1.00146      0.73472  69.53s
#       4       0.86743       0.86594      1.00171      0.73472  69.52s
#       5       0.86658       0.86522      1.00157      0.73472  69.53s
#       6       0.86520       0.86431      1.00103      0.73472  69.53s
#       7       0.86371       0.86350      1.00024      0.73472  69.54s
#       8       0.86223       0.86271      0.99944      0.73472  69.54s
#       9       0.86130       0.86239      0.99874      0.73472  69.54s
#      10       0.86052       0.86195      0.99834      0.73472  69.54s
#      11       0.86019       0.86186      0.99807      0.73472  69.56s
#      12       0.85917       0.86202      0.99669      0.73472  69.57s
#      13       0.85872       0.86154      0.99673      0.73472  69.56s
#      14       0.85884       0.86134      0.99710      0.73472  69.55s
#      15       0.85800       0.86121      0.99627      0.73472  69.55s
#      16       0.85825       0.86097      0.99684      0.73472  69.56s
#      17       0.85766       0.86019      0.99706      0.73472  69.55s
#      18       0.85663       0.86036      0.99566      0.73472  69.54s
#      19       0.85719       0.86509      0.99087      0.73472  69.55s
#      20       0.86032       0.85919      1.00131      0.73472  69.55s
#      21       0.85599       0.85837      0.99722      0.73472  69.59s
#      22       0.85449       0.85756      0.99642      0.73472  69.57s
#      23       0.85435       0.86139      0.99183      0.73472  69.57s
#      24       0.85736       0.85844      0.99874      0.73472  69.56s
#      25       0.85377       0.85976      0.99303      0.73472  69.56s
#      26       0.85334       0.85662      0.99617      0.73472  69.56s
#      27       0.85110       0.85640      0.99381      0.73472  69.56s
#      28       0.84942       0.85646      0.99178      0.73457  69.55s
#      29       0.84826       0.85545      0.99160      0.73457  69.56s
#      30       0.84725       0.85680      0.98886      0.73457  69.57s
#      31       0.84656       0.85706      0.98776      0.73472  69.55s
#      32       0.84620       0.85506      0.98964      0.73457  69.55s
#      33       0.84595       0.85528      0.98908      0.73457  69.54s
#      34       0.84395       0.85564      0.98634      0.73457  69.55s
#      35       0.84257       0.85808      0.98192      0.73457  69.56s
#      36       0.83999       0.85918      0.97766      0.73457  69.56s
#      37       0.83834       0.86006      0.97475      0.73472  69.56s
#      38       0.83617       0.86493      0.96676      0.73443  69.56s
#      39       0.83533       0.85993      0.97140      0.73500  69.56s
#      40       0.83252       0.86406      0.96351      0.73457  69.57s
#      41       0.82859       0.86863      0.95391      0.73443  69.56s
#      42       0.82244       0.87484      0.94011      0.73415  69.57s
#      43       0.81922       0.87496      0.93630      0.73187  69.59s
#      44       0.81487       0.88169      0.92421      0.73143  69.58s
#      45       0.80963       0.88222      0.91772      0.72929  69.58s
#      46       0.80292       0.88089      0.91149      0.73042  69.58s
#      47       0.79969       0.89250      0.89602      0.72831  69.59s
#      48       0.79234       0.92190      0.85946      0.71776  69.58s
#      49       0.78943       0.89650      0.88057      0.72687  69.58s
#      50       0.78019       0.90597      0.86117      0.72558  69.58s

# Estimating predicted labels vs. actual labels using several measures:

# In[ ]:


#y_pred = net1.predict(X)
#import sklearn.metrics as metrics   
#print(metrics.accuracy_score(y, y_pred))


#  0.9375960826738029

# In[ ]:


#print(metrics.confusion_matrix(y, y_pred))


# array([[25346,    86,   318,    33,    27],
#        [  435,  1966,    34,     2,     6],
#        [  909,    23,  4343,     8,     9],
#        [  134,     7,    20,   706,     6],
#        [  106,     4,    21,     4,   573]])

# In[ ]:


#print(metrics.classification_report(y, y_pred))


# In[ ]:


# Since net1 seems to overfit the train data, we will try to add some robustness by defining net2 to which we added DropoutLayer layers between the existing layers and assigned dropout probabilities to each one of them.

# In[ ]:

# In[ ]:


def predictTest():
    import os
    FTEST_PATH = '/home/cse/Downloads/output'     
    outputpath = '/home/cse/Downloads/output/net5Predictions.csv'
    inputNamesPath = '/home/cse/Downloads/output/test_images_128_128_names_new.csv'
    names = read_csv(os.path.expanduser(inputNamesPath),header=None)  # load pandas dataframe
    names = np.array(names)
    with open(outputpath, "w") as outfile:
        outfile.write("image,1,2,3,grade\n")    
    os.chdir(FTEST_PATH)
    j=0
    for fname in sorted(os.listdir('.'), key=os.path.getmtime):
        FTEST = FTEST_PATH+'/'+fname
        X, _ = load(test=True)
        a_pred = net1.predict(X)
        b_pred = net2.predict(X)
        c_pred = net3.predict(X)
        d_pred = net4.predict(X)

        with open(outputpath, "a") as outfile:
            for i in range(0,X.shape[0]):
                #if i+j < names.shape[0]:
                filename = str(names[i+j][0])
                label = str(a_pred[i])
                label1 = str(b_pred[i])
                label2 = str(c_pred[i])
                
                #from scipy.stats import mode
                #label5=mode([label,label1,label2,label3,label4])
                #print(label5)
                lis=[]
                lis.append(label)
                lis.append(label1)
                lis.append(label2)

                from collections import Counter
                counter = Counter(lis)
                max_count = max(counter.values())
                mode = [k for k,v in counter.items() if v == max_count]
                label5=str(mode)
                label6=label5[2]
                
                if label6 > "0":
                	label5 = str(d_pred[i])
                #print(label5)
                outfile.write( filename +','+label+','+label1+','+label2+','+label5+"\n")
        j=j+X.shape[0]
    os.chdir('../')


# Running the defined method yields our results file:

# In[ ]:


import sys
sys.setrecursionlimit(20000)
predictTest()

