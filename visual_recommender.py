#Intstalling necessary gpu library from tensorflow library
!pip install tensorflow-gpu==1.15.5

#Importing necessary tools from tensorflow to implement Visual Recommender Algorithm
import tensorflow as tf
from tensorflow.nn import l2_loss as loss_function
from tensorflow.keras.initializers import glorot_normal
from tensorflow.train import AdamOptimizer as optim_fn
from tensorflow.train import Saver

print(tf.version)

#Importing dataset
import numpy as np

[train_data, val_data, test_data, Item, user_num, item_num] = np.load('AmazonFashion6ImgPartitioned.npy', allow_pickle=True, encoding='bytes')

# Setting the latent dimensionality - latent dimensionlaity is the output dimensions which has extracted features of the Images from our CNN
dims = 100


#Necessary functions to output the neural network layers for building the CNN
def add_conv_2d_layer(inp_layer, wt_layer, stride):
  return tf.nn.conv2d(inp_layer, wt_layer, [1, stride, stride, 1], padding = 'SAME')

def add_bias_layer(inp_layer, bias_layer):
  return tf.nn.bias_add(inp_layer, bias_layer)

def add_relu(inp_layer):
  return tf.nn.relu(inp_layer)

def add_dropout(inp_layer, dropout):
  return tf.nn.dropout(inp_layer, 0.5)

def add_maxpool_2d_layer(inp_layer):
  return tf.nn.max_pool(inp_layer, [1,2,2,1], [1,2,2,1], padding = 'SAME')

def initialize_wt_layer(variable, shape):
  return tf.compat.v1.get_variable(variable, dtype=tf.float32, shape = shape, initializer=glorot_normal())

def initialize_bias_layer(variable, shape):
  return tf.compat.v1.get_variable(variable, dtype=tf.float32, initializer=tf.zeros([shape]))

def placeholder(dtype, shape):
  return tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=None)

from os import waitid_result
## Weights and Bias vector Initializations 

# For Hidden CNN Layer 1
wtcnn1 = initialize_wt_layer('wt_cnn1', [11, 11, 3, 64])
bcnn1 = initialize_bias_layer('b_cnn1', 64)

# For Hidden CNN Layer 2
wtcnn2 = initialize_wt_layer('wt_cnn2', [5, 5, 64, 256])
bcnn2 = initialize_bias_layer('b_cnn2', 256)

# For Hidden CNN Layer 3
wtcnn3 = initialize_wt_layer('wt_cnn3', [3, 3, 64, 256])
bcnn3 = initialize_bias_layer('b_cnn3', 256)

# For Hidden CNN Layer 4
wtcnn4 = initialize_wt_layer('wt_cnn4', [3, 3, 64, 256])
bcnn4 = initialize_bias_layer('b_cnn4', 256)

# For Hidden CNN Layer 5
wtcnn5 = initialize_wt_layer('wt_cnn5', [3, 3, 64, 256])
bcnn5 = initialize_bias_layer('b_cnn5', 256)

# For Fully Connected Dense Layer 1
wtdl1 = initialize_wt_layer('wt_dl1', [7*7*256, 4096])
bdl1 = initialize_bias_layer('b_dl1', 4096)

# For Fully Connected Dense Layer 2
wtdl2 = initialize_wt_layer('wt_dl2', [4096, 4096])
bdl2 = initialize_bias_layer('b_dl2', 4096)

# For Fully Connected Dense Layer 3
wtdl3 = initialize_wt_layer('wt_dl3', [4096, dims])
bdl3 = initialize_bias_layer('b_dl3', dims)

# Building the CNN Model, with above initialized weight and bias vectors
def CNN_Model(input):
  input = (tf.cast(input, tf.float32) - 127.5)/127.5
  # Input layer reshaping to account for all 3 RGB layers and flattening.
  input = tf.reshape(input, shape = [-1, 224, 224, 3])
  
  # Layer grouping architecture : Conv2d -> bias -> relu -> maxpool
  # We have removed Max pool in two hiddenlayers to account for image blurring as max pool blurs the image much more.

  # Layer 1 
  cnnl1 = add_conv_2d_layer(input, wtcnn1, 4)
  cnnl1 = add_bias_layer(cnnl1, bcnn1)
  cnnl1 = add_relu(cnnl1)
  cnnl1 = add_maxpool_2d_layer(cnnl1)

  # Layer 2
  cnnl2 = add_conv_2d_layer(cnnl1, wtcnn2, 1)
  cnnl2 = add_bias_layer(cnnl2, bcnn2)
  cnnl2 = add_relu(cnnl2)
  cnnl2 = add_maxpool_2d_layer(cnnl2)

  # Layer 3
  cnnl3 = add_conv_2d_layer(cnnl2, wtcnn3, 1)
  cnnl3 = add_bias_layer(cnnl3, bcnn3)
  cnnl3 = add_relu(cnnl3)

  # Layer 4
  cnnl4 = add_conv_2d_layer(cnnl3, wtcnn4, 1)
  cnnl4 = add_bias_layer(cnnl4, bcnn4)
  cnnl4 = add_relu(cnnl4)

  # Layer 5
  cnnl5 = add_conv_2d_layer(cnnl4, wtcnn5, 1)
  cnnl5 = add_bias_layer(cnnl5, bcnn5)
  cnnl5 = add_relu(cnnl5)
  cnnl5 = add_maxpool_2d_layer(cnnl5)

  # Fully Connected Layer 1
  dl1 = tf.reshape(cnnl5, [-1, 7*7*256])
  dl1 = tf.add(tf.matmul(dl1, wtdl1), bdl1)
  dl1 = add_relu(dl1)
  dl1 = add_dropout(dl1, 0.5)

  # Fully Connected Layer 2
  dl2 = tf.add(tf.matmul(dl1, wtdl2), bdl2)
  dl2 = add_relu(dl2)
  dl2 = add_dropout(dl2, 0.5)

  # Fully Connected Layer 3 == Output layer == Dimensions [4096,100]
  opl = tf.add(tf.matmul(dl2, wtdl3), bdl3)
  
  return opl


# We have to take the user preferences and also the images he has positively interacted with
# so we would implement queue based batch training here since the images corpus is very large.
# Enqueue five batches which has user, category, and item and Image vectors respectively.
# Dequeue them and pass them to the CNN for training, loss_function calculation, and Optimization

u_shape = [1]
i_shape = [1]
j_shape = [1]
img_shape = [224,224,3]
batch_size = 128


with tf.device('/GPU:0'):
  # training batchwise
  # we define place holders for each user_number, item_number and Images seperately.
  u_Q = placeholder(tf.int32, u_shape)
  i_Q = placeholder(tf.int32, i_shape)
  j_Q = placeholder(tf.int32, j_shape)
  

  img_Q1 = placeholder(tf.int32, img_shape)
  img_Q2 = placeholder(tf.int32, img_shape)
  
  # We would 'Enqueue' a list of these objects inside a fifoqueue and then dequeue when passing these to the CNN.
  Q_train = tf.queue.FIFOQueue(128*5, [tf.int32,tf.int32,tf.int32,tf.int32,tf.int32], [u_shape,i_shape,j_shape,img_shape,img_shape], None, None, 'fifo_queue')

  # Enqueue Operation
  Q_enqueue = Q_train.enqueue([u_Q,i_Q,j_Q,img_Q1,img_Q2], name=None)

  # Dequeue Operation
  u, i, j, img_1, img_2 = Q_train.dequeue_many(batch_size)

  # Dropout variable to use in CNN, it would help in dropping some unnecessary connections, so that number trainable variables in weights and biases would reduce. 
  dropout_prob = placeholder(tf.float32, None)

  #Reshaping the dequeued items to proper shape to feed it as an input u,i,j has shape [1], dequeue 128 of them youll get a vector of [128,1]
  # Similarly we have image size of [224,224,3] we are putting an object of 128 images in the queue, dequeue the object we have an 128 images, reshape to [128,224,224,3]
  u = tf.reshape(u, shape=[128], name=None)
  i = tf.reshape(i, shape=[128], name=None)
  j = tf.reshape(j, shape=[128], name=None)

  imgtst = placeholder(tf.uint8, [128,224,224,3]) 
  _imgtst = (tf.cast(imgtst, tf.float32) - 127.5)/127.5

  # CNN Training

  # This output of CNN represents the actual recommendations, which will be simultaneoulsy trained with the user latent vector(reg_const)
  # This is the important part of the algorithm, where the extracted features of CNN(image features) will be impacting the user latent vector(reg_const)
  # This way, CNN would optimize the image features, and then the user latent vector( this is an indication of user preference ) would be optimized as well
  with tf.compat.v1.variable_scope('DVBPR') as vr_scope:
    # This is CNN extracted features.
    op1 = CNN_Model(img_1)
    vr_scope.reuse_variables()
    # This image would be replaced by the image created by GAN in the recommendation image geneartion process, this would be key because in loss function, we would 
    # like to reduce the loss value between these two images, so that, CNN image features and the GAN generated image would be similar, 
    # resulting to effective recommendations.
    op2 = CNN_Model(img_2)
    optest = CNN_Model(_imgtst)

    # Regularization to improve the models performance
    wt_regs = [wtdl1, wtdl2, wtdl3, wtcnn1, wtcnn2, wtcnn3, wtcnn4, wtcnn5]
    # loss values of each weight and bias vectors
    nn_regs = sum(map(loss_function, wt_regs))

    # THIS IS THE USER LATENT VECTOR, has dimansions of [(no. of users),100] --> represents the preference/latent vector of each user (hence (user_num * dims ))
    reg_const = tf.Variable(tf.random.uniform([user_num, dims], 0, 1) * 0.01)

  # Loss value and Optimization process
  loss_value = 0
  training_loss = tf.reduce_sum(tf.math.log(tf.math.sigmoid(tf.reduce_sum(tf.math.multiply(tf.gather(reg_const, u), tf.math.subtract(op1, op2)), 1, keepdims=True))))
  regs = loss_function(tf.gather(reg_const, u))
  training_loss = training_loss - (0.001 * nn_regs + 0.99 * regs)
  adam_optim_fn = optim_fn().minimize(training_loss)

import random

#AUC ROC Calculation 
def auc_roc_score(train_data, test_data, user_data, item_data):
  auc_score = 0
  classes = 0
  for obj in train_data:
    item = test_data[obj][0]['productid']
    temp = np.dot(user_data[obj,:],item_data.temp)
    classes += 1
    product_set = []
    for it in train_data[obj]:
      product_set.add(it['productid'])
    product_set.add(item)

    correct_preds = 0
    each_item_preds = 0
    for j in random.sample(range(item_num),100):
      if temp[i]>temp[j]: each_item_preds +=1
      correct_preds += 1
    each_item_preds = each_item_preds/correct_preds
    ans += each_item_preds
  auc_score/=float(classes)

  return auc_score

from PIL import Image
from io import StringIO
# Loading stringified image into the correct image sizes [224,224,3] to input in training queues.
def image_as_nparray(input):
  image_load = Image.open(StringIO(Item[input]['imgs'])).convert('RGB')
  img_array = np.asarray(image_load.resize(224,224))
  return np.uint8(img_array)

#Evaluation of the model, 
def model_eval(step):
  I= np.zeros([item_num, dims], np.float32)
  J= session.run(reg_const)
  index = np.array_split(range(item_num),(item_num+127)/128)

  inp_imgs = np.zeros([128,224,224,3], np.int32)
  length = len(index)
  for item in range(length):
    temp = 0
    for i in index[item]:
      inp_imgs[temp] = image_as_nparray(i)
      temp += 1
    I[index[item][0]:(index[i][-1]+1)] = session.run(optest, feed_dict = {imgtst:inp_imgs})[:(index[item][-1] - index[i[0] + 1])]
  np.save('Batch_'+str(dims)+'Eval_step_'+str(step)+'.npy',[J, I])

  return auc_roc_score(train_data, val_data, J, I), auc_roc_score(train_data, test_data, J, I)

# picking the user, item category sample form dasatset to put in queue for training
def user_sample_picking(user):
  usr = random.randrange(user_num)
  num_usr = len(user[usr])
  item = user[usr][random.randrange(num_usr)]['productid']
  temp = []
  for i in user[usr]:
    temp.add(i['productid'])
  while True:
    j = random.randrange(item_num)
    if (not j in temp): break
  return (usr, item, j)

# Loading the five items u, i, j, img_1, img_2 into the training queue.
def image_loading():
  while True:
    (user, item, j) = user_sample_picking(train_data)
    image1 = image_as_nparray(item)
    image2 = image_as_nparray(j)
    session.run(Q_enqueue, feed_dict={ u_Q:np.asarray([user]), i_Q:np.asarray([item]), j_Q:np.asarray([j]), img_Q1:image1, img_Q2:image2 })

# Define tensorflow sessions, and configurations.
from tensorflow import Session, ConfigProto

session = Session(config = ConfigProto(allow_soft_placement=True, log_device_placement=False))
session.run(tf.initialize_all_variables())

# The Actual training starts here, the variables and functions defined in the training process would be run using the tensorflow session defined. 
iterator_length = 0
steps_so_far = 1 
for i in train_data:
  iterator_length += len(train_data[i])

save_model = Saver([i for i in tf.compat.v1.global_variables() if i.name.startswith('VisRec')])

no_of_epochs = 0
epochs_to_train = 10
while steps_so_far * 128 <= epochs_to_train * iterator_length+1 :
  session.run(adam_optim_fn, feed_dict={dropout_prob : 0.5})

  if steps_so_far * 128 > no_of_epochs * iterator_length:
    no_of_epochs += 1
    # Save your trained model in a checkpoint file, for every epoch the optimized model would be stored.
    save_model.save(session, './Visual_Recommender_Evaluation_'+str(dims)+'.ckpt')
    validation_auc_score, test_auc_score = model_eval(steps_so_far)
    print('For Epoch {}: Validation AUC Score == {} & Test AUC Score == {}'.format(str(no_of_epochs), str(validation_auc_score), str(test_auc_score)))
  steps_so_far += 1