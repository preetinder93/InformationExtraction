import pandas as pd
import json

def getTrainingSet(csvPath):
    data = pd.read_csv(csvPath)

    taggedData = data.iloc[:,:].values

    #print(data.head(3))

    #trainingRowMask = '["*text*",{"entities":[[*recStart*,*recEnd*,"RECIPIENT"],[*givStart*,*givEnd*,"GIVER"],[*relStart*,*relEnd*,"RELATIONSHIP"],[*streetStart*,*streetEnd*,"STREET"],[*cityStart*,*cityEnd*,"CITY"],[*provStart*,*provEnd*,"PROVINCE"],[*postalStart*,*postalEnd*,"POSTAL"]]}]'
    trainingRowMask = '["*text*",{"entities":[[*corporation_start*,*corporation_end*,"corporation"],[*corporation_number_start*,*corporation_number_start*,"corporation_number"],[*file_date_start*,*file_date_end*,"file_date"]]}]'
    trainingSet = []
    for row in taggedData:
        #tempRow = trainingRowMask.replace('*text*',str(row[0])).replace('*recStart*',str(row[2])).replace('*recEnd*',str(row[3])).replace('*givStart*',str(row[5])).replace('*givEnd*',str(row[6])).replace('*relStart*',str(row[8])).replace('*relEnd*',str(row[9])).replace('*streetStart*',str(row[11])).replace('*streetEnd*',str(row[12])).replace('*cityStart*',str(row[14])).replace('*cityEnd*',str(row[15])).replace('*provStart*',str(row[17])).replace('*provEnd*',str(row[18])).replace('*postalStart*',str(row[20])).replace('*postalEnd*',str(row[21]))        
        tempRow = trainingRowMask.replace('*text*',str(row[0])).replace('*corporation_start*',str(row[2])).replace('*corporation_end*',str(row[3])).replace('*corporation_number_start*',str(row[5])).replace('*corporation_number_end*',str(row[6])).replace('*file_date_start*',str(row[8])).replace('*file_date_end*',str(row[9]))
        #print(str(row[0]))
        tempRowJson = json.loads(tempRow)
        trainingSet.append(tempRowJson)
    
    #print(trainingSet)
    #print('='*80)
    return trainingSet
        

#getTrainingSet("C:\\Users\\preet\\workspace-neon\\InformationExtraction\\Images\\OCROutput\\taggedData2.csv")



