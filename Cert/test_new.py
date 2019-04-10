import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

DATASET_PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Cert\\animal_data\\'
CHECKPOINT_PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Cert\\animal_data\\tmp'

train_dataset = h5py.File(DATASET_PATH + 'train_animals.h5', "r")
#train_dataset = h5py.File('test_signs.h5', "r")
ls=list(train_dataset.keys()) #muestra etiquetas de h5
print(ls) 


train_x = np.array(train_dataset["train_set_x"]) # your train set features
train_y = np.array(train_dataset["train_set_y"]) # your train set labels
max(train_y)

test_dataset = h5py.File(DATASET_PATH + 'test_animals.h5', "r")
test_x = np.array(test_dataset["test_set_x"]) # your train set features
test_y = np.array(test_dataset["test_set_y"]) # your train set labels

print(train_x.shape)
print(train_y.shape)

# Hyperparameters
training_epochs=2
batch_size=16  
learning_rate = 0.001
display_step=1

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Crea los placeholders para la sesión.
    
    Argumentos:
    n_H0 -- Escalar, height de la imagen de entrada
    n_W0 -- Escalar, width de la imagen de entrada
    n_C0 -- Escalar, Número de canales de entrada
    n_y -- Escalar, Número de clases
        

    Returna:
    X -- placeholder para los datos de entrada, de tamaño [None, n_H0, n_W0, n_C0] y dtype "float"
    Y -- placeholder para las etiquetas de entrada, de tamaño [None, n_y] y dtype "float"
    """

    #### Haga su código acá ### (≈2 lines)

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    return X, Y

def initialize_parameters():
    """
    Inicializa los parámetros (Pesos) para construir la red neuronal convolucional con tensorflow. El tamaño es
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
                        W3 : [1, 1, 16, 32]
                        
    usar: tf.get_variable("W1", [, , , ], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    Returna:
    parameters -- Un diccionario de tensores que contiene W1, W2
    """
    
    tf.set_random_seed(1)                              # 
        
    #### Haga su código acá ### (≈2 lines)
        
    W1 = tf.get_variable("W1", [3, 3, 3, 24], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [3, 3, 24, 24], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W3", [3, 3, 24, 24], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable("W4", [3, 3, 24, 24], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    ### Fin ###

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implementa la propagación hacia adelante del modelo
    CONV2D -> RELU -> CONV2D -> RELU -> MAXPOOL  -> CONV2D -> RELU -> CONV2D -> RELU -> MAXPOOL >FULLYCONNECTED
    
    
    #CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED -> FULLYCONNECTED
    
    Argumentos:
    X -- placeholder de entrada (ejemplos de entrenamiento), de tamaño (input size, number of examples)
    parameters -- Diccionario que contiene los parámetros "W1", "W2" desde initialize_parameters

    Returna:
    Z3 -- Salida de la última unidad LINEAR 
    """
    
    # Obtención de los pesos desde "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters ['W4']
    
    
    # CONV2D: stride of 1, padding 'SAME'
    
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')

    # RELU
    
    A1 = tf.nn.relu(Z1)
    
    # CONV2D: stride of 1, padding 'SAME'
    
    Z2 = tf.nn.conv2d(X, W2, strides = [1,1,1,1], padding = 'SAME')
    
    # RELU
    
    A2 = tf.nn.relu(Z2)
    
    
    # MAXPOOL: window 2x2, stride 1, padding 'SAME'
    
    P1 = tf.nn.max_pool(A2, ksize= [1,2,2,1], strides = [1,1,1,1], padding = 'SAME')
    
    # CONV2D: stride of 1, padding 'SAME'
    
    Z3 = tf.nn.conv2d(X, W3, strides = [1,1,1,1], padding = 'SAME')

    # RELU
    
    A3 = tf.nn.relu(Z3)
    
    # CONV2D: stride of 1, padding 'SAME'
    
    Z4 = tf.nn.conv2d(X, W4, strides = [1,1,1,1], padding = 'SAME')
    
    # RELU
    
    A4 = tf.nn.relu(Z4)
    
    
    # MAXPOOL: window 2x2, stride 1, padding 'SAME'
    
    P2 = tf.nn.max_pool(A4, ksize= [1,2,2,1], strides = [1,1,1,1], padding = 'SAME')
    
    
     # FLATTEN    
      
    F = tf.contrib.layers.flatten(P2)
    print(F)
    
    #capa fullconnect
    FC = tf.contrib.layers.fully_connected(F,21,None)
 

    
    
    #Z5 = tf.contrib.layers.fully_connected(A4,21,None) # como es la capa de salina el # de neuronas es el # de clases
    return FC


def compute_cost(FC, Y):
    """
    Calcula la función de costo
    
    Argumentos:
    Z4 -- Salida del forward propagation (Salida de la última unidad LINEAR), de tamaño (6, Número de ejemplos)
    Y -- placeholders con el vector de etiquetas "true", del mismo tamaño que Z3

    Returns:
    cost - Tensor de la función de costo
    """
    
    #### Haga su código acá ### (≈2 lines)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = FC, labels = Y))
    
    ### Fin ###
    
    return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    #print(m)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.001, num_epochs=2, batch_size=16, print_cost = True):
    # ops.reset_default_graph()                         # Permite correr nuevamente el modelo sin sobreescribir las tf variables
    tf.set_random_seed(1)                             #  (tensorflow seed)
    seed = 3                                          # 
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = [] 
    times = []     
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    
    # Inicializar Parámetros
    
    parameters = initialize_parameters()
    
    # Forward propagation: Construir el forward propagation en el grafo de tensorflow
    
    FC = forward_propagation(X, parameters)
        
    # Cost function: Incluir la  función de costo en el grafo de tensorflow
        
    cost = compute_cost(FC, Y)
    
    # Backpropagation: Define el optimizador. Usar AdamOptimizer para minimizar el costo.

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # Inicializar todas las variables globales
    
    init = tf.global_variables_initializer()
    
    # genera el objeto para guardar el modelo entrenado
    
    saver = tf.train.Saver()
     
    # Iniciar la sesión 
    with tf.Session() as sess:
        
        # Run init
        sess.run(init)
        
        # Loop de entrenamiento
        for epoch in range(num_epochs):

            batch_size_cost = 0.
            num_batch_size = int(m / batch_size) # número de minibatches de tamaño minibatch_size en el conjunto de entrenamiento          seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, batch_size, seed) #se revuelve tanto x como y
            

            for minibatch in minibatches:

                # Seleccionar un minibatch

                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).

                _ , temp_cost = sess.run([optimizer, cost], {X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / batch_size
                

            # Imprime el costo
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                minutes = ((time.time() - start_time))
                minutes = minutes/float(60)
                times.append(("Duration: %.2f Minutes" % minutes))
                print("Duration: %.2f Minutes" % minutes)
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # Graficar la función de costo
        plt.figure(2)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calcular las predicciones correctas
        predict_op = tf.nn.softmax(FC)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(predict_op, 1), tf.argmax(Y, 1)) 
        
        # Calcular la predicción sobre el conjunto de test 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        minutes = ((time.time() - start_time))
        minutes = minutes/float(60)
        times.append(("Final Duration: %.2f Minutes" % minutes))
        print("Final Duration: %.2f Minutes" % minutes)
        
       
        return train_accuracy, test_accuracy, parameters
    
_, _, parameters = model(train_x, train_y, test_x, test_y)
