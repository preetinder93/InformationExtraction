#from OCRImage import getOCROutput
import spacy
import pandas as pd
from GenerateTrainingDataFromDict import getTrainingSet
from GetCoordinates import generateTaggedData
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example
import time
from spacy.language import Language
from spacy.tokens import Span
import json

#generate tagged data
generateTaggedData("C:\\Python\\Training\\GIT\\InformationExtraction\\Images\\OCROutput\\OCRData.csv")
print("TaggedData Ready")
print('*'*80)

#prepare Training Set
TRAIN_DATA = getTrainingSet("C:\\Python\\Training\\GIT\\InformationExtraction\\Images\\OCROutput\\taggedData6.csv")
print('*'*80)
##print(type(train))
print(TRAIN_DATA[0])

nlp = spacy.blank("en")
print("*** Pipe names in NLP model = ", nlp.pipe_names)
if "ner" not in nlp.pipe_names:
    #ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner", last=True)
    ner = nlp.get_pipe("ner")
    
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for itn in range(2):
        modelName = 'C:\\Python\\Training\\GIT\\InformationExtraction\\TrainedModelLibrary\\TrainedModelIteration' + str(itn)
        print("Running Iteration = ", str(itn))
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            #print("annotations = ",annotations)
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.2,sgd=optimizer,losses=losses)
        print(losses)
        nlp.to_disk(modelName)
                
