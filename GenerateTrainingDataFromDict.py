import pandas as pd
import json

def getTrainingSet(csvPath):
    data = pd.read_csv(csvPath)

    taggedData = data.iloc[:,:].values

    print(data.head(3))

    TRAIN_DATA = []

    for dataRow in taggedData:
        entities = []
        entities.append((dataRow[2],dataRow[3],"corporation_name"))
        entities.append((dataRow[5],dataRow[6],"corporation_num"))
        entities.append((dataRow[8],dataRow[9],"file_date"))
        TRAIN_ROW = [dataRow[0],{"entities": entities}]
        #print(TRAIN_ROW)
        TRAIN_DATA.append(TRAIN_ROW)

    fileName = "C:\\Python\\Training\\GIT\\InformationExtraction\\Images\\OCROutput\\training_dataV3.json"
    with open(fileName, "w", encoding="utf-8") as f:
        json.dump(TRAIN_DATA, f, indent=4)

    return TRAIN_DATA
    
    
        

#getTrainingSet("C:\\Python\\Training\\GIT\\InformationExtraction\\Images\\OCROutput\\taggedData_corp_numV2.csv")



