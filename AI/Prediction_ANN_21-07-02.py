import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
# changed keras.callbacks.callbacks to keras.callbacks
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import PReLU
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

# changed read_excel to read_csv
data = pd.read_csv("C:/Users/Fredrica Holgersson/Desktop/AI/Natanael_4.csv")
data_norm = (data.iloc[:,1:5]-np.min(data.iloc[:,1:5]))/(np.max(data.iloc[:,1:5])-np.min(data.iloc[:,1:5]))
data_min = np.array(np.min(data.iloc[:,1:4])).reshape(1,-1)
data_max = np.array(np.max(data.iloc[:,1:4])).reshape(1,-1)
#mark = np.loadtxt("C:/Users/Fredrica Holgersson/Desktop/AI/Natanael_4.txt")
#mark = data
#mark[:,[0, 1]] = mark[:,[1, 0]]
#mark_norm =(mark-data_min)/(data_max-data_min)
X_data = np.array(data_norm.iloc[:,0:3])
y_data = np.array(data_norm.iloc[:,3])






# =============================================================================
# define the parameters
activation = 'tanh'
num_neuron = 15
optimizer = 'rmsprop'


def ANN(num_neuron,optimizer,activation,HL1,X_data,y_data,iteration,WA):
    def plt_fitness(y_train,pred_train):
        plt.figure(figsize = (12,10))
        plt.subplot(2,2,1)
        plt.scatter(y_train,pred_train)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.title('Whole data, R2_score: ' + str('%.2f' % r2_score(y_train,pred_train))+'|  Variance_score:  '+str('%.2f' % explained_variance_score(y_train,pred_train)))
        plt.show()
        
    if optimizer  == 'adam':
        optim = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    if  optimizer == 'rmsprop':
        optim = optimizers.RMSprop(learning_rate=0.01, rho=0.9)
    if  optimizer == 'nadam':
        optim = optimizers.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    # =============================================================================
    # # create model and save results
    data_mse = pd.DataFrame(columns=['Run #0', 'Run #1', 'Run #2','Average','Std','Avg +1 Std','Avg -1 Std']) 
                                   
    def TwoHiddenLayer_BuildModel(num_neuron,HL1):
         #alpha = 10
         model = Sequential()
         model.add(Dense(HL1, input_dim=3, kernel_initializer='RandomNormal', bias_initializer='zeros',activation=activation))
         model.add(Dense(num_neuron-HL1, input_dim=3, kernel_initializer='RandomNormal', bias_initializer='zeros',activation=activation))
         model.add(Dense(1,kernel_initializer='RandomNormal', bias_initializer='zeros',activation=activation))
#         model.add(Dense(HL1, input_dim=3, kernel_initializer='RandomNormal', bias_initializer='zeros'))
#         model.add(PReLU())
#         model.add(Dense(num_neuron-HL1, input_dim=3, kernel_initializer='RandomNormal', bias_initializer='zeros'))
#         model.add(PReLU())
#         model.add(Dense(1,kernel_initializer='RandomNormal', bias_initializer='zeros',activation = 'linear'))
         return model
     
        
    
    path_model = "C:/Users/Fredrica Holgersson/Desktop/AI/models"

    
    for i in np.arange(3):
        model = TwoHiddenLayer_BuildModel(num_neuron,HL1)
        HL2 =num_neuron-HL1
        
        path_name = path_model+'/WholeDatasetFiltered_2HL-'+optimizer+'-'+str(num_neuron)+'-'+str(HL1)+'-'+str(HL2)+'-'+activation+'/'
    
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=optim, metrics=['mae','mape'])
        # checkpoint
        if not os.path.isdir(path_name):
               os.mkdir(path_name)
        if not os.path.isdir(path_name+'Runing_'+str(i)+'/'):
               os.mkdir(path_name+'Runing_'+str(i)+'/')
        filepath=path_name+'Runing_'+str(i)+'/{epoch}-{val_loss:.2f}.h5'
    #    es = EarlyStopping(monitor='loss', mode='min', min_delta = 0.01,verbose=1, patience=2)
    #    mc = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='auto')
        callbacks_list = [checkpoint]   
        history = model.fit(X_data, y_data, epochs=iteration,validation_data= (X_data,y_data), verbose=2,callbacks=callbacks_list)
            # =============================================================================
        # #create new arrays to stor results for 3 times running
        if i == 0: 
           data_mse['Run #0'] =  history.history['loss']
        if i == 1:
           data_mse['Run #1'] =  history.history['loss']
        if i == 2: 
           data_mse['Run #2'] =  history.history['loss']
           data_mse['Average'] =  np.mean(data_mse.iloc[:,0:4],axis = 1)
           data_mse['Std'] = np.std(data_mse.iloc[:,0:4],axis = 1)
           data_mse['Avg +1 Std'] = data_mse['Average']+data_mse['Std']
           data_mse['Avg -1 Std'] = data_mse['Average']-data_mse['Std']
    # =============================================================================
    # # save resluts (MSE for dataing and validating)
    MSE = pd.DataFrame(columns=['Data', '+1 Std', '-1 Std'])
    MSE['Dataing'] = history.history['loss']
    MSE['+1 Std'] = MSE['Data']+np.std(MSE['Data'])
    MSE['-1 Std'] = MSE['Data']-np.std(MSE['Data'])
    # =============================================================================
    # Checck which epoch in three runs has the best model
    min_mse_data = np.min(data_mse.iloc[:,0:3],axis = 0)
    num_run = np.array(np.where (min_mse_data == min(min_mse_data))).ravel()
    ep_num = np.array(np.where(data_mse.iloc[:,num_run]==np.min(data_mse.iloc[:,num_run]))).ravel()
    # =============================================================================
    # use the best model to predict on data,validation and test data
    path = path_name+'Runing_'+str(int(num_run))+'/'+str(ep_num[0])+'*'
    file = glob.glob(path)[0]
    
    best_model = TwoHiddenLayer_BuildModel(num_neuron,HL1)
    best_model.load_weights(file)
    # Compile model
    best_model.compile(loss='mean_squared_error', optimizer=optim, metrics=['mae','mape'])
    pred_data= best_model.predict(X_data,verbose = 1)
    plt.scatter(y_data, pred_data)
    line = [max(min(y_data), min(pred_data)), min(max(y_data), max(pred_data))]
    #plt.plot(line, line)
    plt.plot([0,1],[0,1])

    #pred_mark= best_model.predict(mark_norm,verbose = 1) 
# =============================================================================
#     save data,test, validate and corresponding data
    #true_pred_data = pd.DataFrame(columns=['data', 'pred_data'])
    #true_pred_data['data'] = Nataneal
    #true_pred_data['pred_data'] = pred_data*(max(WA)-min(WA))+min(WA)
# =============================================================================
   # plt_fitness(true_pred_data.data,true_pred_data.pred_data)
    plt.savefig(path_name+'Fitness_BestModel.png',dpi = 100,bbox_inches = 'tight')
  # =============================================================================
    #Natanael= pd.DataFrame(columns=['pred_WA'])
    #Natanael['pred_Nata'] = pred_mark.ravel()*(max(WA)-min(WA))+min(WA)
# =============================================================================
    # write out results
    #writer = pd.ExcelWriter(path_name+'Report.txt', engine='writer')
    #MSE.to_excel(writer,sheet_name = 'MSE single report')
    #true_pred_data.to_excel(writer,sheet_name  ='True_pred_data')
    #writer.save()
    #np.savetxt(path_model+'WholeDatasetFiltered_2HL-'+optimizer+'-'+str(num_neuron)+'-'+str(HL1)+'-'+str(HL2)+'-'+activation+'/'+'Natanael_4.csv',WA)




ANN(num_neuron, optimizer, activation, 10, X_data, y_data, 100, data["WA"])




