import pandas as pd
import numpy as np
import base64
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from random import randrange
import paho.mqtt.client as mqtt
import time
import sys
import json
import ast

borker_address = "192.168.0.174"
borker_port = 1883
print("Connecting broker is ", borker_address, borker_port)
keep_alive = 8000
topic_train = "train"
topic_aggregate = "aggregate"

#This function is used fro dataset generation
def InitilizeClients():
    metric = {'accuracy' : [], 'loss' : []}
    index = {'index' : 0}

    with open('data/indexFile.txt', "w") as f:
        f.write(json.dumps(index))
    with open('data/metrics.txt', "w") as f:
        f.write(json.dumps(metric))
    with open('data/localMetrics.txt', "w") as f:
        f.write(json.dumps(metric))

    print("Devices initilization done")

def saveLearntMetrice(file_name,score):

    with open(file_name,'r+') as f:
        trainMetrics = json.load(f)
        trainMetrics['accuracy'].append(score[1])
        trainMetrics['loss'].append(score[0])
        f.seek(0) 
        f.truncate()
        f.write(json.dumps(trainMetrics))

"Training module"
def train(payload):
    print("Starting training...")
    payload_dict = eval(payload)
    epochs = payload_dict['epochs']

    continousTrainingBatchSize = payload_dict['batch']
    print("Batch size is ", continousTrainingBatchSize)
    with open("data/current_model.h5","wb") as file:
        file.write(base64.b64decode(payload_dict['model_file']))
    model = load_model("data/current_model.h5")
    # model.summary()

    #Reading index to simulate continous learning
    currentIndex = 0
    with open('data/indexFile.txt', "r+") as f:
        fileIndex = json.load(f)
        currentIndex = fileIndex['index']

    print("Current Index is ", currentIndex)

    data = pd.read_csv('data/data.csv')
    
    totalRowCount = data.shape[0]
    nextIndex = currentIndex + continousTrainingBatchSize if currentIndex + continousTrainingBatchSize < totalRowCount else totalRowCount
    X = data.iloc[currentIndex:nextIndex,1:-1].values
    y = data.iloc[currentIndex:nextIndex,-1].values
    y = to_categorical(y)

    #print("Dimension of current data ", X.shape)

    #Updating Index
    if nextIndex == totalRowCount:
        nextIndex = 0
    with open('data/indexFile.txt', "w") as f: 
        index = {'index' : nextIndex}
        f.write(json.dumps(index))


    #Printing aggregated global model metrics
    score = model.evaluate(X, y, verbose=0)
    print("Global model loss : {} Global model accuracy : {}".format(score[0], score[1]))
    
    saveLearntMetrice('data/metrics.txt', score)

    try :
        model.fit(X, y, batch_size=32, epochs=epochs, shuffle=True, verbose=0)
    except Exception as e:
        print(e)
        print("Error in training in current iteration")
        return payload_dict['model_file']  
           
    #Printing loss and accuracy after training 
    score = model.evaluate(X, y, verbose=0)
    print("Local model loss : {} Local model accuracy : {}".format(score[0], score[1]))
    
    saveLearntMetrice('data/localMetrics.txt', score)

    # #Save current model 
    model.save('data/model.h5')
    with open('data/model.h5','rb') as file:
        encoded_string = base64.b64encode(file.read())
    
    print("Local training completed")
    return encoded_string

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(topic_train, 0)
    else:
        print("Failed to connect, return code ", rc)

def on_message(client, userdata, msg):
        if msg.topic == topic_train:
            param = train(msg.payload)
            client.publish(topic_aggregate, param)



def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscriptioin done")
    client.publish("connected", "I am connected", 2)

mqttc = mqtt.Client()  
mqttc.will_set("disconnected", "LOST_CONNECTION", 0, False)
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.on_subscribe = on_subscribe

mqttc.username_pw_set("client","smart")
mqttc.connect(borker_address, borker_port, keep_alive)
InitilizeClients()
mqttc.loop_forever()
