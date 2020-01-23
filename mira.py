# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        "Iterating over the iterations and the instances"
        high_acc = 0
        for C in Cgrid:
            self.initializeWeightsToZero()
            for iterations in range(self.max_iterations):
                for i in range(len(trainingData)):
                    #input data
                    instance = trainingData[i]
                    #the correct label corresponding with the data
                    correct_label = trainingLabels[i]
                    #the classified label
                    instance_guess = max(self.classify([instance]))
                    if not instance_guess == correct_label:
                        "The Tau calculations"

                        x = self.weights[instance_guess].__sub__(self.weights[correct_label])
                        y = x.__mul__(instance)
                        y = y + 1.0
                        length = instance.totalCount()
                        z = 2 * (length)
                        if(z == 0):
                            r = C
                        else:
                            float_div = float(y / z)
                            r = min([C, float_div])

                        "multiply instance by r"
                        temp_instance = instance.copy()
                        for key,value in instance.iteritems():
                            temp_instance[key] = value * r

                        #update weights
                        self.weights[correct_label] =   self.weights[correct_label].__add__(temp_instance)
                        self.weights[instance_guess] = self.weights[instance_guess].__sub__(temp_instance)

            #weights are set, now we need to eval to find best C values
            for i in range(len(validationData)):
                instance = validationData[i]
                correct_label = validationLabels[i]
                guessed_label = max(self.classify([instance]))
                correct = 0
                false = 0
                if(correct_label == guessed_label):
                    correct += 1
                else:
                    false += 1
                acc = correct/(correct+false)
                #if this C value yields a higher accuracy, save these weights
                if(acc > high_acc):
                    best_weights = self.weights
        #set weights to the weights of the best C-value
        self.weights = best_weights


    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses
