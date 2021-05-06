import pandas as pd
import re

def getCoordinates(text,pattern):
    #print('Inside getCoordinates function')
    startIndex = text.find(pattern)
    endIndex = startIndex + len(pattern)
    return startIndex, endIndex

def generateTaggedData(csvPath):
    #print('Inside generateTaggedData function')
    data = pd.read_csv(csvPath)

    columnNames = data.iloc[:,1:]##if CSV has ID then make this to 2
    taggedData = []
    columnsList = []
    
    for index, row in data.iterrows():

        text = row['text'].replace('\n', ' ').replace('\r', '').replace('\x0C','').replace('!@#$%^&*()[]{};:,./<>?\|`~-=_+', ' ').replace('\\','').replace('\"','')

        #Add Text value and Text column in respective lists
        rowData = [text]
        columns = ['text']
        
        for column in columnNames:
            
            startIndex, endIndex = getCoordinates(text,row[column])

            #Append each column data and its start and end indices
            rowData.append(row[column])
            rowData.append(startIndex)
            rowData.append(endIndex)

            #Append column name and names of its start and end indices column names
            columns.append(column)
            columns.append(column+"_start")
            columns.append(column+"_end")
            
        columnsList = columns
        taggedData.append(rowData)
        
    df = pd.DataFrame(taggedData, columns = [column for column in columnsList])
    
    df.to_csv("C:\\Python\\Training\\GIT\\InformationExtraction\\Images\\OCROutput\\taggedData6.csv", sep=',', index=False)

#generateTaggedData('C:\\Python\\Training\\GIT\\InformationExtraction\\Images\\OCROutput\\OCRData.csv')
