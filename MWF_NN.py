# import data:
# x_test      -> Testing input data (M,12) 
# x_train     -> Training input data (N,12) 
# y_test      -> Testing output data (M,1)
# y_train     -> Training output data (N,1)
# rayleigh_B  -> Scale parameter of the rayleigh distribution fit to x_train

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input, Add, PReLU
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# Define custom loss function
def custom_loss_function(y_actual,y_pred):
  Rayleigh_dist = (y_actual / K.square(rayleigh_B)) * K.exp(-(K.square(y_actual))/(2*K.square(rayleigh_B)))
  Rayleigh_max = (rayleigh_B / K.square(rayleigh_B)) * K.exp(-(K.square(rayleigh_B))/(2*K.square(rayleigh_B)))
  weight =  0.8 * ((-1 * Rayleigh_dist) + Rayleigh_max) / Rayleigh_max + 0.2
  custom_loss_value = weight*K.mean(K.square((y_actual-y_pred)))

  return custom_loss_value 

# Define custom score for the gridsearch
custom_scorer = make_scorer(custom_loss_function, greater_is_better=False)

# Build residual network architecture
def build_residual_neural_network(optimizer, nodes, layers, residual):
    input_shape = x_test[1].shape

    X_input = Input(input_shape)

    X = Dense(12,input_dim=12,kernel_initializer = 'glorot_uniform',kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00))(X_input)
    X = Activation(PReLU())(X)
    X_shortcut = X

    for nlayer in range(1,layers+1):
        X = Dense(nodes,kernel_initializer = 'glorot_uniform',kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00))(X)
        X = Activation(PReLU())(X)

    X_shortcut = Dense(1,kernel_initializer = 'glorot_uniform',kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00))(X_shortcut)
    X_shortcut = Activation(PReLU())(X_shortcut)    

    X = Dense(1,kernel_initializer = 'glorot_uniform',kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00))(X)

    if residual:
        X = Add()([X,X_shortcut])

    X = Activation(PReLU())(X)

    model = Model(inputs = X_input, outputs = X)

    model.compile(optimizer=optimizer,
         loss=custom_loss_function,
         metrics=['mean_absolute_error','mean_squared_error'])

    return model

# perform 5-fold cross validation grid search to estimate the optimal hyper parameters
estimator = KerasRegressor(build_fn=build_residual_neural_network)

parameters ={'batch_size':[8],
            'nb_epoch':[100],
            'optimizer':['adam'],
            'nodes':[20,30,40,50,60],
            'layers':[3,4,5,6,7],
            'residual':[True]}

gridSearch = GridSearchCV(estimator=estimator,
             param_grid=parameters,
             n_jobs=4,
             cv=5,
             return_train_score=True,
             scoring=custom_scorer)            

grid_result = gridSearch.fit(x_test, y_test)

best_params = grid_result.best_params_

# Build and run the final model
model = build_residual_neural_network(best_params["optimizer"],best_params["nodes"],best_params["layers"],best_params["residual"])

epochs = 100
batch_size = 8

model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(x_train, y_train,
          validation_split=0.2,           
          batch_size=batch_size,
          callbacks=[callback],
          epochs=epochs,
          shuffle=True)

predictions = model.predict([x_test])
