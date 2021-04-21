import pandas as pd
import json

def getTrainingSet(csvPath):
    data = pd.read_csv(csvPath)

    taggedData = data.iloc[:,:].values

    #print(data.head(3))

    trainingRowMask = '["*text*",{"entities":[[*recStart*,*recEnd*,"RECIPIENT"],[*givStart*,*givEnd*,"GIVER"],[*relStart*,*relEnd*,"RELATIONSHIP"],[*streetStart*,*streetEnd*,"STREET"],[*cityStart*,*cityEnd*,"CITY"],[*provStart*,*provEnd*,"PROVINCE"],[*postalStart*,*postalEnd*,"POSTAL"]]}]'
    trainingSet = []
    trainingRows = ''
    for row in taggedData:
        tempRow = trainingRowMask.replace('*text*',str(row[0])).replace('*recStart*',str(row[2])).replace('*recEnd*',str(row[3])).replace('*givStart*',str(row[5])).replace('*givEnd*',str(row[6])).replace('*relStart*',str(row[8])).replace('*relEnd*',str(row[9])).replace('*streetStart*',str(row[11])).replace('*streetEnd*',str(row[12])).replace('*cityStart*',str(row[14])).replace('*cityEnd*',str(row[15])).replace('*provStart*',str(row[17])).replace('*provEnd*',str(row[18])).replace('*postalStart*',str(row[20])).replace('*postalEnd*',str(row[21]))
        #print(type(tempRow))
        #print(tempRow)
        tempRowJson = json.loads(tempRow)
        #print(type(tempRowJson))
        #print(tempRowJson)
        trainingSet.append(tempRowJson)
        #trainingRows = trainingRows + ',' + tempRow
    #trainingRows = trainingRows.replace(',','',1)
    print(type(trainingSet))
    print(trainingSet)
##    print(trainingRows)
##    print('='*80)
    return trainingSet
        

#getTrainingSet("C:\\Python\\Training\\ML\\InformationExtraction\\Images\\OCROutput\\taggedData2.csv")



