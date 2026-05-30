
#!pip install Keras-Preprocessing
import numpy as np
import pandas as pd
import nltk

import tensorflow as tf
from nltk.corpus import stopwords
from nltk.corpus import reuters
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import joblib
from collections import Counter
from textblob import Word 
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Activation, Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Flatten, GRU, Conv1D, MaxPooling1D, Bidirectional
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests
from keras.regularizers import l2
#pip install ktrain
import ktrain
from ktrain import text
sns.set()
#matplotlib inline
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('gutenberg')
nltk.download('brown')
nltk.download("reuters")
nltk.download('words')

df=pd.read_csv("dataset/Suicide_Methods_Dataset1.csv", engine='python', encoding='UTF-8')
df=df.replace('Potential Suicide Method Post ','Potential Suicide Method Post')
print(df)

df['Sentence']=df['Sentence'].fillna("") 
df.isna().sum()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#Convert to lower case
df['lower_case']= df['Sentence'].apply(lambda x: x.lower())   
#Tokenize
tokenizer = RegexpTokenizer(r'\w+')
df['Special_word'] = df.apply(lambda row: tokenizer.tokenize(row['lower_case']), axis=1)  
#Stop words remove
stop = stopwords.words('english')
stop.remove("not")
stop.remove("here")
stop.remove("some")
df['stop_words'] = df['Special_word'].apply(lambda x: [item for item in x if item not in stop])
df['stop_words'] = df['stop_words'].astype('str')
#Filter words based on length
df['short_word'] = df['stop_words'].str.findall('\w{3,}')
df['string']=df['short_word'].str.join(' ') 
#Removing non-english words(mention,emoji,link,special characters etc..)
words = set(nltk.corpus.words.words())
for w in reuters.words():
  words.add(w)
for w in brown.words():
  words.add(w)
for w in gutenberg.words():
  words.add(w)
df['NonEnglish'] = df['string'].apply(lambda x: " ".join(x for x in x.split() if x in words))  
#Lemmatization
df['tweet'] = df['NonEnglish'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

print(df.head(5))

#/home/ubuntu/ml/twitter-suicidal-ideation-detection/Notebook/machinelearning.py

# ## **Applying N-gram**
print ('Applying N-gram*')
x_train, x_test, y_train, y_test = train_test_split(df["Sentence"],df["Label"], test_size = 0.25, random_state = 42)    
count_vect = CountVectorizer(ngram_range=(1, 2))        
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

print (x_train_tfidf.shape,x_test_tfidf.shape, y_train.shape, y_test.shape)

# # **Machine Learning Models**
# # **Logistic Regression**
print ('Logistic Regression')
lr = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
lr.fit(x_train_tfidf, y_train)
y_pred1 = lr.predict(x_test_tfidf)
print("Accuracy: "+str(accuracy_score(y_test,y_pred1)))
print(classification_report(y_test, y_pred1))

scores = cross_val_score(lr, x_train_tfidf,y_train, cv=5)  
print(accuracy_score(y_test,y_pred1))
print ("Cross-validated scores:", scores)

# # **Support Vector Machine**
print ('Support Vector Machine')
svc = LinearSVC()
svc.fit(x_train_tfidf, y_train)
y_pred2 = svc.predict(x_test_tfidf)
print("Accuracy: "+str(accuracy_score(y_test,y_pred2)))
print(classification_report(y_test, y_pred2))

scores = cross_val_score(svc, x_train_tfidf,y_train, cv=5)   
print(accuracy_score(y_test,y_pred2))
print ("Cross-validated scores:", scores)

joblib.dump(svc, 'Suicide_SVM.pkl')
#model = joblib.load('Suicide_SVM.pkl')

## **Naive Bayes(Multinomial)**
print ('Naive Bayes(Multinomial)') 
mnb = MultinomialNB()
mnb.fit(x_train_tfidf, y_train)
y_pred3 = mnb.predict(x_test_tfidf)
print("Accuracy: "+str(accuracy_score(y_test,y_pred3)))
print(classification_report(y_test, y_pred3))

scores = cross_val_score(mnb, x_train_tfidf,y_train, cv=5)   
print(accuracy_score(y_test,y_pred3))
print ("Cross-validated scores:", scores)

# ## **Randomforest**
# print ('Randomforest') 
# rfc = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
# rfc.fit(x_train_tfidf,y_train)
# y_pred4 = rfc.predict(x_test_tfidf)
# print("Accuracy: "+str(accuracy_score(y_test,y_pred4)))
# print(classification_report(y_test, y_pred4))

# scores = cross_val_score(rfc, x_train_tfidf,y_train, cv=5)   
# print(accuracy_score(y_test,y_pred4))
# print ("Cross-validated scores:", scores)

# # **GradientBoosting Classifier**
# print ('GradientBoosting Classifier') 
# gbc = GradientBoostingClassifier(n_estimators=1000, max_features='auto', max_depth=4, random_state=1, verbose=1)
# gbc.fit(x_train_tfidf, y_train)
# y_pred5 = gbc.predict(x_test_tfidf)
# print(accuracy_score(y_test, y_pred5))
# print(classification_report(y_test, y_pred5))

# scores = cross_val_score(gbc, x_train_tfidf,y_train, cv=5)   
# print(accuracy_score(y_test,y_pred5))
# print ("Cross-validated scores:", scores)

# # ## **Ensemble Classifier**
# print ('Ensemble Classifier') 
# mnb = MultinomialNB()
# rfc= RandomForestClassifier(n_estimators=1000, max_depth=12, random_state=42)
# lr = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
# svc = SVC(probability=True)
# ec=VotingClassifier(estimators=[('Multinominal NB', mnb), ('Random Forest', rfc),('Logistic Regression',lr),('Support Vector Machine',svc)], voting='soft', weights=[1,2,3,4]) 
# ec.fit(x_train_tfidf,y_train)                                                 
# y_pred6 = ec.predict(x_test_tfidf)
# print(accuracy_score(y_test, y_pred6))
# print(classification_report(y_test, y_pred6))

# scores = cross_val_score(ec, x_train_tfidf,y_train, cv=5)  
# print(accuracy_score(y_test,y_pred6))
# print ("Cross-validated scores:", scores)

# mc = count_vect.transform([' i will kill myself'])
# m = transformer.transform(mc)
# y_pred = ec.predict(m)
# print(y_pred)

# joblib.dump(ec, 'Suicide_Ensemble.pkl')
# #model = joblib.load('Suicide_Ensemble.pkl')

# # ## **AdaBoost with Random Forest Classifier**
# print ('AdaBoost with Random Forest Classifier') 
# rfc = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)
# abc= AdaBoostClassifier(base_estimator=rfc, learning_rate=0.2, n_estimators=100)
# abc.fit(x_train_tfidf, y_train)                                                   
# y_pred7= abc.predict(x_test_tfidf)
# print("Accuracy: "+str(accuracy_score(y_test, y_pred7)))
# print(classification_report(y_test, y_pred7))

# scores = cross_val_score(abc, x_train_tfidf,y_train, cv=5)   
# print(accuracy_score(y_test,y_pred7))
# print ("Cross-validated scores:", scores)

# # # **Comparison Between ML Models**
# print ('Comparison Between ML Models') 
# Comparison_unibi = pd.DataFrame({'Logistic Regression': [accuracy_score(y_test,y_pred1)*100,f1_score(y_test,y_pred1,average='macro')*100,recall_score(y_test, y_pred1,average='micro')*100,precision_score(y_test, y_pred1,average='micro')*100],
#                             'SVM':[accuracy_score(y_test,y_pred2)*100,f1_score(y_test,y_pred2,average='macro')*100,recall_score(y_test, y_pred2,average='micro')*100,precision_score(y_test, y_pred2,average='micro')*100],
#                            'Naive Bayes':[accuracy_score(y_test,y_pred3)*100,f1_score(y_test,y_pred3,average='macro')*100,recall_score(y_test, y_pred3,average='micro')*100,precision_score(y_test, y_pred3,average='micro')*100],
#                            'Random Forest':[accuracy_score(y_test,y_pred4)*100,f1_score(y_test,y_pred4,average='macro')*100,recall_score(y_test, y_pred4,average='micro')*100,precision_score(y_test, y_pred4,average='micro')*100],
#                            'GradientBoosting':[accuracy_score(y_test,y_pred5)*100,f1_score(y_test,y_pred5,average='macro')*100,recall_score(y_test, y_pred5,average='micro')*100,precision_score(y_test, y_pred5,average='micro')*100],
#                            'Ensembled':[accuracy_score(y_test,y_pred6)*100,f1_score(y_test,y_pred6,average='macro')*100,recall_score(y_test, y_pred6,average='micro')*100,precision_score(y_test, y_pred6,average='micro')*100],
#                            'Adaboost':[accuracy_score(y_test,y_pred7)*100,f1_score(y_test,y_pred7,average='macro')*100,recall_score(y_test, y_pred7,average='micro')*100,precision_score(y_test, y_pred7,average='micro')*100],

# })

# print ('Comparison using uni-gram(1,1)') 
# Comparison_unibi.rename(index={0:'Accuracy',1:'F1_score', 2: 'Recall',3:'Precision'}, inplace=True)
# print(Comparison_unibi.head())

# print ('Comparison using bi-gram(2,2)') 
# Comparison_unibi.rename(index={0:'Accuracy',1:'F1_score', 2: 'Recall',3:'Precision'}, inplace=True)
# print(Comparison_unibi.head())

# print ('Comparison using uni-bi-gram(1,2)') 
# Comparison_unibi.rename(index={0:'Accuracy',1:'F1_score', 2: 'Recall',3:'Precision'}, inplace=True)
# print(Comparison_unibi.head())


# # **Deep Learning Models**
print('**Deep Learning Models**')

vocabulary_size =6000
max_text_len = 60
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['Sentence'].values)
le=len(tokenizer.word_index)+1
print(le)
sequences = tokenizer.texts_to_sequences(df['Sentence'].values)
X_DeepLearning = pad_sequences(sequences, maxlen=max_text_len)
X_DeepLearning.shape[1]

# Save the tokenizer object
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

df.loc[df['Label'] == 'Suicide Method Post' , 'LABEL'] = 1     
df.loc[df['Label'] == 'Non Suicide Method Post', 'LABEL'] = 0             
     
labels = to_categorical(df['LABEL'], num_classes=2)
print(labels[:])

XX_train, XX_test, y_train, y_test = train_test_split(X_DeepLearning , labels, test_size=0.25, random_state=42)
print((XX_train.shape, y_train.shape, XX_test.shape, y_test.shape))

# # **LSTM 1-Layer**
print('# **LSTM 1-Layer**')
epochs = 10
emb_dim = 120
batch_size = 50       
model_lstm1 = Sequential()
model_lstm1.add(Embedding(vocabulary_size,emb_dim, input_length=X_DeepLearning.shape[1]))
model_lstm1.add(SpatialDropout1D(0.8))                                             
model_lstm1.add(Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5)))                 
model_lstm1.add(Dropout(0.5))
model_lstm1.add(Flatten())
model_lstm1.add(Dense(64, activation='relu'))
model_lstm1.add(Dropout(0.5))
model_lstm1.add(Dense(2, activation='softmax'))
model_lstm1.compile(optimizer=tf.optimizers.Adam(),loss='binary_crossentropy', metrics=['acc']) 
print(model_lstm1.summary()) 

checkpoint_callback = ModelCheckpoint(filepath="content/lastm-1-layer-best_model.h5", save_best_only=True, monitor="val_acc", mode="max", verbose=1)

early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)

reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)

callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]

history_lstm1 = model_lstm1.fit(XX_train, y_train, epochs = epochs, batch_size = batch_size, validation_split=0.1, callbacks=callbacks)

results_1 = model_lstm1.evaluate(XX_test, y_test, verbose=False)
print(f'Test results - Loss: {results_1[0]} - Accuracy: {100*results_1[1]}%')

# acc = history_lstm1.history['acc']                        
# val_acc = history_lstm1.history['val_acc']
# loss = history_lstm1.history['loss']
# val_loss = history_lstm1.history['val_loss']
# plt.plot( acc, 'go', label='Train accuracy')
# plt.plot( val_acc, 'g', label='Validate accuracy')
# plt.title('Train and validate accuracy')
# plt.legend()                                           

# plt.figure()
# plt.plot( loss, 'go', label='Train loss')
# plt.plot( val_loss, 'g', label='Validate loss')
# plt.title('Train and validate loss')
# plt.legend()
# plt.show()

# Load tokenizer object
with open('content/tokenizer.pickle', 'rb') as handle:
    tokenizers = pickle.load(handle)

model = load_model('content/lastm-1-layer-best_model.h5')
#model.save('/content/drive/MyDrive/Colab_Notebooks/DL Model/Twitter Suicide Ideation Detection/lstm 1-layer.h5') 

twt = ['i will not kill myself']
twt = tokenizers.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=60, dtype='int32')

predicted = model.predict(twt,batch_size=1,verbose = True)
if(np.argmax(predicted) == 0):
    print("Potential Suicide Method Post")
elif (np.argmax(predicted) == 1):
    print("Non Suicide Method Post")

# ## **LSTM 2-Layers**   
epochs = 10
emb_dim = 120                     
batch_size = 50                
model_lstm2 = Sequential()            
model_lstm2.add(Embedding(vocabulary_size,emb_dim ,input_length=X_DeepLearning.shape[1]))
model_lstm2.add(SpatialDropout1D(0.8))
model_lstm2.add(Bidirectional(LSTM(200, dropout=0.5, recurrent_dropout=0.5, return_sequences= True)))
model_lstm2.add(Dropout(0.5))
model_lstm2.add(Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout =0.5)))
model_lstm2.add(Dropout(0.5))
model_lstm2.add(Flatten())
model_lstm2.add(Dense(64, activation='relu'))
model_lstm2.add(Dropout(0.5))
model_lstm2.add(Dense(2, activation='softmax'))
model_lstm2.compile(optimizer=tf.optimizers.Adam(),loss='binary_crossentropy', metrics=['acc']) 
print(model_lstm2.summary())  

checkpoint_callback = ModelCheckpoint(filepath="content/lastm-2-layer-best_model.h5", save_best_only=True, monitor="val_acc", mode="max", verbose=1)

early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)

reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)

callbacks2=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]

history_lstm2 = model_lstm2.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=callbacks2)

results_2 = model_lstm2.evaluate(XX_test, y_test, verbose=False)
print(f'Test results - Loss: {results_2[0]} - Accuracy: {100*results_2[1]}%')

# acc = history_lstm2.history['acc']                          
# val_acc = history_lstm2.history['val_acc']
# loss = history_lstm2.history['loss']
# val_loss = history_lstm2.history['val_loss']

# plt.plot( acc, 'go', label='Train accuracy')
# plt.plot( val_acc, 'g', label='Validate accuracy')
# plt.title('Train and validate accuracy')
# plt.legend()                                            

# plt.figure()

# plt.plot( loss, 'go', label='Train loss')
# plt.plot( val_loss, 'g', label='Validate loss')
# plt.title('Train and validate loss')
# plt.legend()

# plt.show() 

# Load tokenizer object
with open('content/tokenizer.pickle', 'rb') as handle:
    tokenizers = pickle.load(handle)

model = load_model('content/lastm-2-layer-best_model.h5')
#model.save('/content/drive/MyDrive/Colab_Notebooks/DL Model/Twitter Suicide Ideation Detection/lstm 2-layer.h5') 

twt = ["i will not kill myself. "]
twt = tokenizers.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=60, dtype='int32')

predicted = model.predict(twt,batch_size=1,verbose = True)
if(np.argmax(predicted) == 0):
    print("Potential Suicide Method Post")
elif (np.argmax(predicted) == 1):
    print("Non Suicide Method Post")

# ## **GRU**
print('**GRU**')    
epochs = 10
emb_dim = 120                     
batch_size = 50                
model_gru = Sequential()            
model_gru.add(Embedding(vocabulary_size,emb_dim ,input_length=X_DeepLearning.shape[1]))
model_gru.add(SpatialDropout1D(0.5))
model_gru.add(GRU(units=16,  dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)))
model_gru.add(Dropout(0.5))
model_gru.add(Dense(228, activation='relu', kernel_regularizer=l2(0.01)))
model_gru.add(Dropout(0.5))
model_gru.add(Dense(2, activation='softmax'))
model_gru.compile(optimizer=tf.optimizers.Adam(),loss='binary_crossentropy', metrics=['acc']) 
print(model_gru.summary())

checkpoint_callback = ModelCheckpoint(filepath="content/gru-best_model.h5", save_best_only=True, monitor="val_acc", mode="max", verbose=1)

early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)

reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)

callbacks3=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]

history_gru = model_gru.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1, callbacks=callbacks3)

results_3 = model_gru.evaluate(XX_test, y_test, verbose=False)
print(f'Test results - Loss: {results_3[0]} - Accuracy: {100*results_3[1]}%')

# acc = history_gru.history['acc']                          
# val_acc = history_gru.history['val_acc']
# loss = history_gru.history['loss']
# val_loss = history_gru.history['val_loss']

# plt.plot( acc, 'go', label='Train accuracy')
# plt.plot( val_acc, 'g', label='Validate accuracy')
# plt.title('Train and validate accuracy')
# plt.legend()                                            

# plt.figure()

# plt.plot( loss, 'go', label='Train loss')
# plt.plot( val_loss, 'g', label='Validate loss')
# plt.title('Train and validate loss')
# plt.legend()

# plt.show() 

# Load tokenizer object
with open('content/tokenizer.pickle', 'rb') as handle:
    tokenizers = pickle.load(handle)

model = load_model('content/gru-best_model.h5')
#model.save('/content/drive/MyDrive/Colab_Notebooks/DL Model/Twitter Suicide Ideation Detection/gru-best_model.h5') 

twt = ["i will not kill myself."]
twt = tokenizers.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=60, dtype='int32')

predicted = model.predict(twt,batch_size=1,verbose = True)
if(np.argmax(predicted) == 0):
    print("Potential Suicide Method Post")
elif (np.argmax(predicted) == 1):
    print("Non Suicide Method Post")


### **CNN+LSTM**
print('## **CNN+LSTM**')
epochs = 10
emb_dim = 120                                                                
batch_size = 50
model_cl = Sequential()
model_cl.add(Embedding(vocabulary_size,emb_dim, input_length=X_DeepLearning.shape[1]))
model_cl.add(SpatialDropout1D(0.8))
model_cl.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
model_cl.add(MaxPooling1D(pool_size=2))
model_cl.add(Conv1D(filters=32, kernel_size=6, activation='relu'))
model_cl.add(MaxPooling1D(pool_size=2))
model_cl.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model_cl.add(Dropout(0.5))
model_cl.add(Bidirectional(LSTM(400, dropout=0.5, recurrent_dropout=0.5)))
model_cl.add(Dropout(0.5))
model_cl.add(Flatten())
model_cl.add(Dense(64, activation='relu'))
model_cl.add(Dropout(0.5))
model_cl.add(Dense(2, activation='softmax'))
model_cl.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc'])
print(model_cl.summary())     

checkpoint_callback = ModelCheckpoint(filepath="content/cnn+lastm-best_model.h5", save_best_only=True, monitor="val_acc", mode="max", verbose=1)

early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)

reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)

callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]

history_cl = model_cl.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1, callbacks=callbacks)

results_4 = model_cl.evaluate(XX_test, y_test, verbose=False)
print(f'Test results - Loss: {results_4[0]} - Accuracy: {100*results_4[1]}%')

# acc = history_cl.history['acc']                          
# val_acc = history_cl.history['val_acc']
# loss = history_cl.history['loss']
# val_loss = history_cl.history['val_loss']
# plt.plot( acc, 'go', label='Train accuracy')
# plt.plot( val_acc, 'g', label='Validate accuracy')
# plt.title('Train and validate accuracy')
# plt.legend()                                           

# plt.figure()
# plt.plot( loss, 'go', label='Train loss')
# plt.plot( val_loss, 'g', label='Validate loss')
# plt.title('Train and validate loss')
# plt.legend()
# plt.show() 


# Load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizers = pickle.load(handle)
    
model = load_model('content/cnn+lastm-best_model.h5')
model_cl.save('content/CNN+LSTM.h5') 

twt = ['I will not kill myself']
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=60, dtype='int32')

predicted = model.predict(twt,batch_size=1,verbose = True)
if(np.argmax(predicted) == 0):
    print("Potential Suicide Method Post")
elif (np.argmax(predicted) == 1):
    print("Non Suicide Method Post")


# ## **Model Comparision**
print('Model Comparision')
results=pd.DataFrame({'Model':['LSTM-1 Layer','LSTM-2 Layer','GRU','CNN+LSTM'],
                     'Accuracy Score':[results_1[1],results_2[1],results_3[1],results_4[1]]})
result_df=results.sort_values(by='Accuracy Score', ascending=False)
result_df=result_df.set_index('Model')
print(result_df)    

# ## **Bert Model**
print('**Bert Model**')
X_train, X_test, y_train, y_test = train_test_split(df['Sentence'], df['Label'], test_size=0.33, random_state=42)
X_train = X_train.tolist()
X_test = X_test.tolist()
y_train = y_train.tolist()
y_test = y_test.tolist()

class_names = ['Suicide Method Post', 'Non Suicide Method Post']

(x_train,y_train), (x_val,y_val), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                                       x_test=X_test, y_test=y_test,
                                                                       class_names=class_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=140, 
                                                                       max_features=5000)

model = text.text_classifier('bert', train_data=(x_train,y_train), preproc=preproc)                                                                       

learner = ktrain.get_learner(model, train_data=(x_train,y_train), 
                             val_data=(x_val,y_val),
                             batch_size=16)

learner.fit_onecycle(2e-5, 3)

learner.plot()

learner.validate(val_data=(x_val,y_val), class_names=class_names)

predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()

message = 'i will not kill myself by hanging'
prediction = predictor.predict(message)
print('predicted: {}'.format(prediction))

print('**Save Bert Model**')
predictor.save("content/bert_model_Suicide")

# **Load Saved Model and Predict**
print('**Load Saved Model and Predict**')
predictor1 = ktrain.load_predictor('content/bert_model_Suicide')

data = "I'm so tired of pretending that everything is okay. I just want to hang myself."
print(predictor1.predict(data))

