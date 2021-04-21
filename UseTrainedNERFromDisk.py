import spacy
from spacy.lang.en import English
import pandas as pd


#model = English().from_disk("C:\Python\Training\ML\InformationExtraction\TrainedModelLibrary\TrainedModelV1")
model = spacy.load("C:\\Users\\preet\\workspace-neon\\InformationExtraction\\TrainedModelLibrary\\TrainedModelV8")
#model = spacy.load("C:\Python\Training\ML\InformationExtraction\TrainedModelLibrary\modelV2")
print(model.pipe_names)

text = 'TD Canada Trust Gift Letter  To: The Manager TD Canada Trust  ‘This is to continn that the undersigned is making a pilt of $  ‘To: ANHAD SINGH HANJRA Name(s): RABAAB KAUR Relationship: SISTER  Relationship:     ee  Property To Be Purchased:  Street: 700 HUMBERWOOD BLVD City: ETOBICOKE Prov; ONTARIO postal Code: M9W 7J4  No part of the gill is being provided by any Third Party having anv interest (direct or indirect) in the sale of the subjeet property. ‘The monev is a genuine gift and does not have to be repaid.'
##data = pd.read_csv("C:\\Python\\Training\\ML\\InformationExtraction\\Images\\OCROutput\\taggedData2.csv")
##
##X=data['Text']
##
##print(X[0])
##text = X[0]

doc = model(text)

print(doc.ents)

for ent in doc.ents:
    print(ent.text,ent.label_)
