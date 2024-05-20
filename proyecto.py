import sys
import getopt
import os
import math
import collections
import csv
import re
from io import StringIO

class NaiveBayes():
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self, lista):
        """NaiveBayes initialization"""
        self.datos = []
        with open('data/datos.csv', 'rb') as csvfile:
            contenido = csvfile.read()
            reader = csv.reader(csvfile)
            # Filtra los bytes no deseados (0x9d en este caso)
            contenido_filtrado = contenido.replace(b'\x9d', b'')
            
            # Convierte los bytes filtrados de nuevo a una cadena
            contenido_filtrado_str = contenido_filtrado.decode('latin-1')

            # Crea un objeto StringIO para simular un archivo de texto
            csvfile_str = StringIO(contenido_filtrado_str)
            reader = csv.reader(csvfile_str)
            [datos.append(fila) for fila in reader]
                
            
        self.FILTER_STOP_WORDS = False
        self.stopList = set(self.readFile('./data/english.stop'))
        self.numFolds = 10
        self.vocab = set([])
        self.tematicas = list(lista)
        self.prior = [0.0 for i in range(len(lista))]
        self.diccionariosdetematicas = {}
        for elem in lista:
            self.diccionariosdetematicas[elem] = collections.defaultdict(lambda:0)

    def classify(self, words):

        total_count_tematicas = []
        log_prob_tematicas = []
        for elem in self.tematicas:
            suma = sum(self.diccionariosdetematicas[elem].values())
            total_count_tematicas.append(suma)
            log_prob_tematicas.append(0.0)
        # Calculate the total vocabulary size
        vocab_size = len(self.vocab)

        # Add 1 to the total count to account for Laplace smoothing
        for elem in total_count_tematicas:
            elem += vocab_size

        # Calculate log probabilities for both classes simultaneously
        for word in words:
            # Calculate log probability for romance class
            for i in range(len(log_prob_tematicas)):
                log_prob_tematicas[i] += math.log((self.diccionariosdetematicas[self.tematicas[i]][word] + 1) / total_count_tematicas[i])

        # Return the classification based on which class has higher probability
        maximo = -float('inf')
        tematicaRetorna = ''
        for i in range(len(log_prob_tematicas)):
            if maximo < log_prob_tematicas[i]:
                maximo = log_prob_tematicas[i]
                tematicaRetorna = self.tematicas[i]
        return tematicaRetorna

    def addExample(self, klass, words):
        """
        Train the model on an example document with label klass ('pos' or 'neg') and
        words, a list of strings.
        """
        self.vocab.update(words)
        for word in words:
            self.diccionariosdetematicas[klass][word] = self.diccionariosdetematicas[klass][word] + 1
        
    def readFile(self, fileName):
        """
        Code for reading a file. You probably don't want to modify anything here, 
        unless you don't like the way we segment files.
        """
        with open(fileName) as f:
            contents=f.readlines()
        return '\n'.join(contents).split()

    def train(self, split):
        for example in split.train:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words =  self.filterStopWords(words)
            self.addExample(example.klass, words)

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for example in split.test:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words =  self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels

    def altBuildSplits(self, dic):
        """Builds the splits for training/testing"""
        splits = []
        print('[INFO]\tPerforming %d-fold cross-validation on data set' % (self.numFolds))
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for tematica in tematicas:
                for i in range(len(dic[tematica])):
                    example = self.Example()
                    example.words = dic[tematica][i][1].split(' ')
                    example.klass = dic[tematica][i][2]
                    if round(i / 100) == fold:
                        split.test.append(example)
                    else:
                        split.train.append(example)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        [filtered.append(word) for word in words if not word in self.stopList and word.strip() != '']              
        return filtered

if __name__ == "__main__":

    dic = collections.defaultdict(lambda: [])
    datos = []
    avgAccuracy = 0.0
    fold = 0

    print("Welcome to the Naive-Bayes lyric theme classificator!")

    with open('data/datos.csv', 'rb') as csvfile:
        contenido = csvfile.read()
        reader = csv.reader(csvfile)
        # Filtra los bytes no deseados (0x9d en este caso)
        contenido_filtrado = contenido.replace(b'\x9d', b'')
        # Convierte los bytes filtrados de nuevo a una cadena
        contenido_filtrado_str = contenido_filtrado.decode('latin-1')   
        # Crea un objeto StringIO para simular un archivo de texto
        csvfile_str = StringIO(contenido_filtrado_str)
        reader = csv.reader(csvfile_str)
        [datos.append(fila) for fila in reader if fila[2] != 'topic']
                     
    todasTematicas = []
    [todasTematicas.append(fila[2]) for fila in datos]
        
    tematicas = set(todasTematicas)
    nb = NaiveBayes(tematicas)
    nb.FILTER_STOP_WORDS = True  # Set to True to remove stop words

    [dic[fila[2]].append(fila) for fila in datos]

    splits = nb.altBuildSplits(dic)

    for split in splits:
        classifier = NaiveBayes(tematicas)
        accuracy = 0.0
        for example in split.train:
            words = example.words
            if nb.FILTER_STOP_WORDS:
                words = classifier.filterStopWords(words)
            classifier.addExample(example.klass, words)

        for example in split.test:
            words = example.words
            if nb.FILTER_STOP_WORDS:
                words = classifier.filterStopWords(words)
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy))
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print('[INFO]\tAccuracy: %f' % avgAccuracy)

    print("If you want to classify some lyrics press '1', if you wanna exit the program type any other character")
    userInput = input()
    while userInput=='1':
        print('Type in your lyrics:')
        lyricInput = input()
        lyrics = re.findall(r'\w+|[^\w\s]', lyricInput)
        guess = classifier.classify(set(lyrics))
        print("Lyrics classified as:", guess)
        userInput = '1'
