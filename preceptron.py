#I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else. I understand that if I am involved in plagiarism or cheating I will have to sign an official form that I have cheated and that this form will be stored in my official university record. I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that I will receive a grade of “F” for the course for any additional offense.
#author: Bo Zhang

#!/usr/bin/env python

import os
import sys
import collections
import re
import copy

#stopwords dir
#stopwords_dir="stopwords.txt"
#train spam dir
#spam_train_dir = "train/spam/"
#train ham dir
#ham_train_dir = "train/ham/"
#test spam dir
#spam_test_dir = "test/spam/"
#test ham dir
#ham_test_dir = "test/ham/"


#store the email instance
class Document:
    text=""
    word_frequence={}

    true_class=""
    learned_class=""

    def __init__(self,text,counter,true_class):
        self.text=text
        self.word_frequence=counter
        self.true_class=true_class

    def getText(self):
        return self.text

    def getWordFrequence(self):
        return self.word_frequence;

    def getLearningClass(self):
        return self.learned_class

    def setLearningClass(self,guess):
        self.learned_class=guess

    def getTrueClass(self):
        return self.true_class




def getWords(text):
#    words=re.findall(r'\w+',text)
#    for k in range(len(words)):
#        words[k]=words[k].lower()
#        words[k]=words[k].replace("_","")
#        words[k]=words("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",words[k])
#    new_words=set(words)
    res=collections.Counter(re.findall(r'\w+', text))
    return dict(res)


def read_file(filename):
    with open(filename,'r',encoding='gb18030', errors='ignore') as text_file:
        text = text_file.read()
        #        print(text)
        #        exit()
        return text


def getDataSet(dict1,directory,true_class):
    for entry in os.listdir(directory):
        entry_path=os.path.join(directory,entry)
        if os.path.isfile(entry_path):
            with open(entry_path,'r',encoding='gb18030', errors='ignore') as text_file:
                text=text_file.read()
                dict1.update({entry_path: Document(text,getWords(text),true_class)})

def setStopWord(stop_word_file):
    stops=[]
    with open(stop_word_file,'r',encoding='gb18030', errors='ignore') as file:
        stops=(file.read().splitlines())
    return stops

def removeStopWord(stops,data_set):
    data_set=copy.deepcopy(data_set)
    for i in stops:
        for j in data_set:
            if i in data_set[j].getWordFrequence():
                del data_set[j].getWordFrequence()[i]
    return data_set

#perceptron training
#adjust weight every training set
#iterator all the dataset and update weight
#If >0 , then spam, else ham
def training(weights, learning_rate, training_set, iterations, classes):
    for i in iterations:
        for j in training_set:
            activation=weights['weight_zero']
            for k in training_set[j].getWordFrequence():
                if k not in weights:
                    weights[k]=0.0
                activation+=weights[k]*training_set[j].getWordFrequence()[k]
            res=0.0
            if activation>0:
                res=1.0
            value=0.0
            if training_set[j].getTrueClass()==classes[1]:
                value=1.0
            for w in training_set[j].getWordFrequence():
                weights[w]+=float(learning_rate)*float((value-res))* float(training_set[j].getWordFrequence()[w])
#            print("%d" %weights[w])
#            exit();



#collect all the word
def collectWord(data_set):
    vector=[]
    for i in data_set:
        for j in data_set[i].getWordFrequence():
            if j not in vector:
                vector.append(j)
    return vector


#predict the result
def predict(weights,classes,instance):
    activation=weights['weight_zero']
    for i in instance.getWordFrequence():
        if i not in weights:
            weights[i]=0.0
        activation+=weights[i]*instance.getWordFrequence()[i]
    if activation>0:
        return 1
    else:
        return 0



def main(train_directory, test_directory, iterations,learning_rate):
    training_dataset={}
    test_dataset={}

    filtered_training_dataset={}
    filtered_test_dataset={}

    stop_words=setStopWord('stop_words.txt')

    classes=["ham","spam"]
    
    
    iterations = iterations
    learning_rate = learning_rate

    getDataSet(training_dataset, train_directory + "/spam", classes[1])
    getDataSet(training_dataset, train_directory + "/ham", classes[0])
    
#        for i in training_dataset:
#            print(training_dataset[i].getText(),end=' ')
#            print(training_dataset[i].getWordFrequence())
#        exit()

    getDataSet(test_dataset, test_directory + "/spam", classes[1])
    getDataSet(test_dataset, test_directory + "/ham", classes[0])
   
#    for i in test_dataset:
#        print(test_dataset[i].getText(),end=' ')
#        print(test_dataset[i].getWordFrequence())
#    exit()


    filtered_training_dataset = removeStopWord(stop_words, training_dataset)
    filtered_test_dataset = removeStopWord(stop_words, test_dataset)

    
    training_dataset_vocab = collectWord(training_dataset)
    filtered_training_dataset_vocab = collectWord(filtered_training_dataset)
    
    
    weights = {'weight_zero': 1.0}
    filtered_weights = {'weight_zero': 1.0}
    for i in training_dataset_vocab:
        weights[i] = 0.0
    for i in filtered_training_dataset_vocab:
        filtered_weights[i] = 0.0
#    for key in training_set_words:


#    training

    training(weights,learning_rate,training_dataset,iterations,classes)
#    for i in weights:
#        if weights[i]<0:
#            print(i,end=' ')
#            print(weights[i])
#    exit()
    training(filtered_weights,learning_rate,filtered_training_dataset,iterations,classes)

    #test
    count=0
    for i in test_dataset:
        guess=predict(weights,classes,test_dataset[i])
        if guess==1:
            test_dataset[i].setLearningClass(classes[1])
            if test_dataset[i].getTrueClass() == test_dataset[i].getLearningClass():
                count+=1
        if guess==0:
            test_dataset[i].setLearningClass(classes[0])
            if test_dataset[i].getTrueClass() == test_dataset[i].getLearningClass():
                count+=1

    filt_count=0
    for i in filtered_test_dataset:
        guess=predict(filtered_weights,classes,filtered_test_dataset[i])
        if guess==1:
            filtered_test_dataset[i].setLearningClass(classes[1])
            if filtered_test_dataset[i].getTrueClass() == filtered_test_dataset[i].getLearningClass():
                filt_count+=1
        if guess==0:
            filtered_test_dataset[i].setLearningClass(classes[0])
            if filtered_test_dataset[i].getTrueClass() == filtered_test_dataset[i].getLearningClass():
                filt_count+=1

    print("Result number of correct %d, total number %d" %(count,len(test_dataset)))
    print("Accurate: %.4f%%" %((float(count)/float(len(test_dataset)))*100.0))
    print("Filtered number of correct %d, total number %d" %(filt_count,len(filtered_test_dataset)))
    print("Filtered Accurate: %.4f%%" %((float(filt_count)/float(len(filtered_test_dataset)))*100.0))

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
