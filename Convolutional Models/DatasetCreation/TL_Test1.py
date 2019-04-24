import h5py
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import tensornets as nets

# Hiperparametros
training_epochs=10
batch_size=16  
learning_rate = 0.01
display_step=1

# Parametros de la red neuronal
n_hidden_1 = 1024 # 1st layer number of neurons
n_input = 215 # data input (feature shape ?,7,7,2048)
n_classes = 21 # total classes

X = tf.placeholder(tf.float32, [None, 224, 224, 3])
X2 = tf.placeholder(tf.float32, [None, 14, 14, 512])
Y = tf.placeholder("float", [None, n_classes])


DATA_DIR = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\'



# Leer de una base de datos H5
with h5py.File('animal_data/train_animals.h5','r') as h5data:

    ls=list(h5data.keys())
    print(ls)
    train_data=np.array(h5data.get('train_set_x')[:])
    train_labels=np.array(h5data.get('train_set_y')[:])
print(train_data.shape)    
print(train_labels.shape) 

with h5py.File('animal_data/test_animals.h5','r') as h5data:
    ls=list(h5data.keys())
    print(ls)    
    test_data=np.array(h5data.get('test_set_x')[:])
    test_labels=np.array(h5data.get('test_set_y')[:])
print(test_data.shape)    
print(test_labels.shape)  

# Cargo el modelo VGG16 de Tensornets, modelo con que se trabajara
model = nets.VGG16(X)
assert isinstance(model, tf.Tensor)

# Hago resize a las imagenes para que queden con su arquitectura
def resize_np (np_array):
    resized=[]
    for i in list(np_array):
        larger=cv2.resize(i,(224,224))
        resized.append(np.array(larger))
    return (np.array(resized).astype(np.float32))

# Funcion para cambiar labels a onehot
def one_hot_transformation(labels,n_classes):
    samples=labels.size
    one_hot_labels=np.zeros((samples,n_classes))
    for i in range(samples):
        one_hot_labels[i,labels[i]]=1
    return(one_hot_labels)
  
X_train=resize_np(train_data)
print(X_train.shape)
X_test=resize_np(test_data)
print(X_test.shape)
Y_train=one_hot_transformation(train_labels,n_classes)
print(Y_train.shape)
Y_test=one_hot_transformation(test_labels,n_classes)
print(Y_test.shape)

# Normalizo las imagenes
X_train = X_train/255
X_test = X_test/255


# mostrar un ejemplo
plt.imshow(X_train[20])

# Imprimo el modelo(solo para ver la arquitectura)
model.print_middles()


def initialize_parameters():
  
  tf.set_random_seed(1)

  W1 = tf.get_variable('W1',[3,3,512,512],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
  W2 = tf.get_variable('W2',[3,3,512,512],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
  
  
  parameters = {"W1": W1,
                "W2": W2}
  
  return parameters
parameters = initialize_parameters()


# Declaración de los pesos y los bias
weights = {'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
           
          }
biases  = {'b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
          
          }


def multilayer_perceptron(x, parameters):
    
    # Obtención de los pesos desde "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']

    # 1ra CONV2D, padding 'SAME'
    
    Z1 = tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='SAME')
    
    # RELU 1
    
    A1 = tf.nn.relu(Z1)
    
    # 2da CONV2D, padding 'SAME'
    
    Z2 = tf.nn.conv2d(A1,W2,strides=[1,1,1,1],padding='SAME')
    
    # RELU 2
    
    A2 = tf.nn.relu(Z2)
    
    # MAXPOOL: window 2x2, stride 2 padding 'SAME'
    
    P1 = tf.nn.max_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME');
    
    # FLATTEN
    
    F = tf.contrib.layers.flatten(P1)  
    
    # FC con 21 salidas
    Z7 = tf.contrib.layers.fully_connected(F,21,None)
    
    # RELU 3
    A2 = tf.nn.relu(Z7)
    
    # FC Salida OUTPUT
    FOutput = tf.contrib.layers.fully_connected(A2,21,None)
    
    
    return Z7
  
# Declarar la operación que aplica el MLP usando la información de entrada
logits = multilayer_perceptron(X2, parameters)

# Funcion de perdida y optimización 
# para el entrenamiento.
loss_op = tf.losses.softmax_cross_entropy(
    onehot_labels=Y,
    logits=logits,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
)
# Optimizador Adam
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    print("¡Inicio de Pre-entrenamiento!")
    sess.run(model.pretrained())  # equivalent to nets.pretrained(model
    print("¡Finalización de Pre-entrenamiento!")
    for epoch in range(training_epochs):        
        avg_cost = 0.
        #obtiene el numero de grupos en que queda dividida la base de datos
        total_batch = int(Y_train.shape[0]/batch_size) 
        print("Inicio Epoch:",epoch," For barch: ",total_batch)
        
        # ciclo para entrenar con cada grupo de datos
        losses=[]
        for i in range(total_batch-1):
            batch_x= X_train[i*batch_size:(i+1)*batch_size]
            batch_y= Y_train[i*batch_size:(i+1)*batch_size]
            features = model.preprocess(batch_x)
            features = sess.run(model.get_middles(), {X: batch_x})[-1]
            # Correr la funcion de perdida y la operacion de optimización con la respectiva alimentación del placeholder
            _,c =sess.run([train_op, loss_op],feed_dict={X2:features,Y:batch_y})
            # Promedio de resultados de la funcion de pérdida
            losses.append(c)
            avg_cost += c / total_batch
        # Mostrar el resultado del entrenamiento por grupos
        if epoch % display_step == 0:
            print("     Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("¡Optimización  Finalizada!")
 