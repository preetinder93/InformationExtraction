from OCRImage import getOCROutput
import spacy
import pandas as pd
from GenerateTrainingData import getTrainingSet
from GetCoordinates import generateTaggedData
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example
import time

##strOutput = getOCROutput('C:\Python\Training\ML\InformationExtraction\Images\image.png','string')
##
##print(strOutput)
##print('='*70)

###generate tagged data
##generateTaggedData("C:\\Python\\Training\\ML\\InformationExtraction\\Images\\OCROutput\\OCRData.csv")
##print("TaggedData Ready")
##print('*'*80)
##
#prepare Training Set
train = getTrainingSet("C:\\Users\\preet\\workspace-neon\\InformationExtraction\\Images\\OCROutput\\taggedData2.csv")
##print('*'*80)
##print(type(train))
##print(train)

#train = [["TD Canada Trust\nGift Letter\n\nTo: The Manager\nTD Canada Trust\n\n�This is to continn that the undersigned is making a pilt of $\n\n�To: AMANINDER SINGH HAJRAH\nName[s]: SIMRATPANNU Relationship: SISTER\n\nRelationship:\n\n \n\nee\n\nProperty To Be Purchased:\n\nStreet: 30 GOLDEN SPRINGS DRIVE City: BRAMPTON\nProv; ONTARIO postal Code: LTA 4N4\n\nNo part of the gill is being provided by any Third Party having anv interest [direct or indirect] in the sale of the subjeet property.\n�The monev is a genuine gift and does not have to be repaid.",{"entities":[[254,277,"STREET"],[284,292,"CITY"],[299,306,"PROVINCE"],[320,327,"POSTAL"]]}], ["TD Canada Trust\nGift Letter\n\nTo: The Manager\nTD Canada Trust\n\n�This is to continn that the undersigned is making a pilt of $\n\n�To: PREETINDER SINGH HAZRAH\nName[s]: SONIA SINGH Relationship: SISTER\n\nRelationship:\n\n \n\nee\n\nProperty To Be Purchased:\n\nStreet: 1 RICHMOND ST W City: TORONTO\nProv; ONTARIO postal Code: M1W 4J1\n\nNo part of the gill is being provided by any Third Party having anv interest [direct or indirect] in the sale of the subjeet property.\n�The monev is a genuine gift and does not have to be repaid.",{"entities":[[255,270,"STREET"],[277,284,"CITY"],[291,298,"PROVINCE"],[312,319,"POSTAL"]]}], ["TD Canada Trust\nGift Letter\n\nTo: The Manager\nTD Canada Trust\n\n�This is to continn that the undersigned is making a pilt of $\n\n�To: JIWANJOT BAJWA\nName[s]: PARAMJOT SINGH Relationship: BROTHER\n\nRelationship:\n\n \n\nee\n\nProperty To Be Purchased:\n\nStreet: 100 ADELAIDE STREET City: TORONTO\nProv; ONTARIO postal Code: M1N 2W2\n\nNo part of the gill is being provided by any Third Party having anv interest [direct or indirect] in the sale of the subjeet property.\n�The monev is a genuine gift and does not have to be repaid.",{"entities":[[250,269,"STREET"],[276,283,"CITY"],[290,297,"PROVINCE"],[311,318,"POSTAL"]]}], ["TD Canada Trust\nGift Letter\n\nTo: The Manager\nTD Canada Trust\n\n�This is to continn that the undersigned is making a pilt of $\n\n�To: ROBERT LAU\nName[s]: RICHARD LEE Relationship: FATHER\n\nRelationship:\n\n \n\nee\n\nProperty To Be Purchased:\n\nStreet: 3975 GRAND PARK DR City: MISSISSAUGA\nProv; ONTARIO postal Code: L8A 4J1\n\nNo part of the gill is being provided by any Third Party having anv interest [direct or indirect] in the sale of the subjeet property.\n�The monev is a genuine gift and does not have to be repaid.",{"entities":[[242,260,"STREET"],[267,278,"CITY"],[285,292,"PROVINCE"],[306,313,"POSTAL"]]}], ["TD Canada Trust\nGift Letter\n\nTo: The Manager\nTD Canada Trust\n\n�This is to continn that the undersigned is making a pilt of $\n\n�To: RICKY SINGH\nName[s]: PREET SINGH Relationship: BROTHER\n\nRelationship:\n\n \n\nee\n\nProperty To Be Purchased:\n\nStreet: 574/8 KAHNUWAN ROAD City: NORTH YORK\nProv; ONTARIO postal Code: LTA 4N8\n\nNo part of the gill is being provided by any Third Party having anv interest [direct or indirect] in the sale of the subjeet property.\n�The monev is a genuine gift and does not have to be repaid.",{"entities":[[244,263,"STREET"],[270,280,"CITY"],[287,294,"PROVINCE"],[308,315,"POSTAL"]]}], ["TD Canada Trust\nGift Letter\n\nTo: The Manager\nTD Canada Trust\n\n�This is to continn that the undersigned is making a pilt of $\n\n�To: JASBIR SINGH BAJWA\nName[s]: NEELAMJIT KAUR Relationship: SISTER\n\nRelationship:\n\n \n\nee\n\nProperty To Be Purchased:\n\nStreet: 6011 167A AVE City: EDMONTON\nProv; ALBERTA postal Code: T6W7J4\n\nNo part of the gill is being provided by any Third Party having anv interest [direct or indirect] in the sale of the subjeet property.\n�The monev is a genuine gift and does not have to be repaid.",{"entities":[[253,266,"STREET"],[273,281,"CITY"],[288,295,"PROVINCE"],[309,315,"POSTAL"]]}], ["TD Canada Trust\nGift Letter\n\nTo: The Manager\nTD Canada Trust\n\n�This is to continn that the undersigned is making a pilt of $\n\n�To: MICHAEL\nName[s]: NANCY DREW Relationship: SPOUSE\n\nRelationship:\n\n \n\nee\n\nProperty To Be Purchased:\n\nStreet: 332 GOLDEN AVE City: WINNIPEG\nProv; MANITOBA postal Code: L79 4N9\n\nNo part of the gill is being provided by any Third Party having anv interest [direct or indirect] in the sale of the subjeet property.\n�The monev is a genuine gift and does not have to be repaid.",{"entities":[[238,252,"STREET"],[259,267,"CITY"],[274,282,"PROVINCE"],[296,303,"POSTAL"]]}]]
#train = [["TD Canada Trust Gift Letter  To: The Manager TD Canada Trust  �This is to continn that the undersigned is making a pilt of $  �To: AMANINDER SINGH HAJRAH Name(s): SIMRATPANNU Relationship: SISTER  Relationship:     ee  Property To Be Purchased:  Street: 30 GOLDEN SPRINGS DRIVE City: BRAMPTON Prov; ONTARIO postal Code: LTA 4N4  No part of the gill is being provided by any Third Party having anv interest (direct or indirect) in the sale of the subjeet property. �The monev is a genuine gift and does not have to be repaid.",{"entities":[[131,153,"RECIPIENT"],[163,174,"GIVER"],[189,195,"RELATIONSHIP"],[254,277,"STREET"],[284,292,"CITY"],[299,306,"PROVINCE"],[320,327,"POSTAL"]]}],["TD Canada Trust Gift Letter  To: The Manager TD Canada Trust  �This is to continn that the undersigned is making a pilt of $  �To: PREETINDER SINGH HAZRAH Name(s): SONIA SINGH Relationship: SISTER  Relationship:     ee  Property To Be Purchased:  Street: 1 RICHMOND ST W City: TORONTO Prov; ONTARIO postal Code: M1W 4J1  No part of the gill is being provided by any Third Party having anv interest (direct or indirect) in the sale of the subjeet property. �The monev is a genuine gift and does not have to be repaid.",{"entities":[[131,154,"RECIPIENT"],[164,175,"GIVER"],[190,196,"RELATIONSHIP"],[255,270,"STREET"],[277,284,"CITY"],[291,298,"PROVINCE"],[312,319,"POSTAL"]]}],["TD Canada Trust Gift Letter  To: The Manager TD Canada Trust  �This is to continn that the undersigned is making a pilt of $  �To: JIWANJOT BAJWA Name(s): PARAMJOT SINGH Relationship: BROTHER  Relationship:     ee  Property To Be Purchased:  Street: 100 ADELAIDE STREET City: TORONTO Prov; ONTARIO postal Code: M1N 2W2  No part of the gill is being provided by any Third Party having anv interest (direct or indirect) in the sale of the subjeet property. �The monev is a genuine gift and does not have to be repaid.",{"entities":[[131,145,"RECIPIENT"],[155,169,"GIVER"],[184,191,"RELATIONSHIP"],[250,269,"STREET"],[276,283,"CITY"],[290,297,"PROVINCE"],[311,318,"POSTAL"]]}],["TD Canada Trust Gift Letter  To: The Manager TD Canada Trust  �This is to continn that the undersigned is making a pilt of $  �To: ROBERT LAU Name(s): RICHARD LEE Relationship: FATHER  Relationship:     ee  Property To Be Purchased:  Street: 3975 GRAND PARK DR City: MISSISSAUGA Prov; ONTARIO postal Code: L8A 4J1  No part of the gill is being provided by any Third Party having anv interest (direct or indirect) in the sale of the subjeet property. �The monev is a genuine gift and does not have to be repaid.",{"entities":[[131,141,"RECIPIENT"],[151,162,"GIVER"],[177,183,"RELATIONSHIP"],[242,260,"STREET"],[267,278,"CITY"],[285,292,"PROVINCE"],[306,313,"POSTAL"]]}],["TD Canada Trust Gift Letter  To: The Manager TD Canada Trust  �This is to continn that the undersigned is making a pilt of $  �To: RICKY SINGH Name(s): PREET SINGH Relationship: BROTHER  Relationship:     ee  Property To Be Purchased:  Street: 574/8 KAHNUWAN ROAD City: NORTH YORK Prov; ONTARIO postal Code: LTA 4N8  No part of the gill is being provided by any Third Party having anv interest (direct or indirect) in the sale of the subjeet property. �The monev is a genuine gift and does not have to be repaid.",{"entities":[[131,142,"RECIPIENT"],[152,163,"GIVER"],[178,185,"RELATIONSHIP"],[244,263,"STREET"],[270,280,"CITY"],[287,294,"PROVINCE"],[308,315,"POSTAL"]]}],["TD Canada Trust Gift Letter  To: The Manager TD Canada Trust  �This is to continn that the undersigned is making a pilt of $  �To: JASBIR SINGH BAJWA Name(s): NEELAMJIT KAUR Relationship: SISTER  Relationship:     ee  Property To Be Purchased:  Street: 6011 167A AVE City: EDMONTON Prov; ALBERTA postal Code: T6W7J4  No part of the gill is being provided by any Third Party having anv interest (direct or indirect) in the sale of the subjeet property. �The monev is a genuine gift and does not have to be repaid.",{"entities":[[131,149,"RECIPIENT"],[159,173,"GIVER"],[188,194,"RELATIONSHIP"],[253,266,"STREET"],[273,281,"CITY"],[288,295,"PROVINCE"],[309,315,"POSTAL"]]}],["TD Canada Trust Gift Letter  To: The Manager TD Canada Trust  �This is to continn that the undersigned is making a pilt of $  �To: MICHAEL Name(s): NANCY DREW Relationship: SPOUSE  Relationship:     ee  Property To Be Purchased:  Street: 332 GOLDEN AVE City: WINNIPEG Prov; MANITOBA postal Code: L79 4N9  No part of the gill is being provided by any Third Party having anv interest (direct or indirect) in the sale of the subjeet property. �The monev is a genuine gift and does not have to be repaid.",{"entities":[[131,138,"RECIPIENT"],[148,158,"GIVER"],[173,179,"RELATIONSHIP"],[238,252,"STREET"],[259,267,"CITY"],[274,282,"PROVINCE"],[296,303,"POSTAL"]]}]]
####X = data['Text']
####
nlp = spacy.load("en_core_web_sm")
##print(nlp.pipe_names)
####
####for text in X:
####    doc = nlp(text)
####    for ent in doc.ents:
####        print(ent.text, ent.start_char, ent.end_char, ent.label_)
####    print('#'*70)
##
#training custom model
ner = nlp.get_pipe("ner")
for _, annotations in train:
    for ent in annotations.get("entities"):
        print(ent)
        #ner.entity.add_label(ent[2])
        ner.add_label(ent[2])
disable_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

with nlp.disable_pipes(*disable_pipes):
    optimizer = nlp.resume_training()

    for iteration in range(100):
        print(iteration)
        random.shuffle(train)
        for raw_text, entity_offsets in train:
            #print("Entitiy Offset",entity_offsets)
            doc = nlp.make_doc(raw_text)
            example = Example.from_dict(doc, entity_offsets)
            nlp.update([example], sgd=optimizer)

    nlp.to_disk("C:\\Users\\preet\\workspace-neon\\InformationExtraction\\TrainedModelLibrary\\TrainedModelV9")
t2 = time.localtime()
current_time = time.strftime("%H:%M:%S", t2)
print(current_time)








