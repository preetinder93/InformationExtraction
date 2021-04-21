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

    columnNames = data.iloc[:,2:]
    taggedData = []
    columnsList = []
    
    for index, row in data.iterrows():

        #Add Text value and Text column in respective lists
        rowData = [row['text']]
        columns = ['text']
        
        for column in columnNames:
            startIndex, endIndex = getCoordinates(row['text'],row[column])

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
    
    df.to_csv("C:\\Users\\preet\\workspace-neon\\InformationExtraction\\Images\\OCROutput\\taggedData2.csv", sep=',', index=False)

generateTaggedData('C:\\Users\\preet\\workspace-neon\\InformationExtraction\\Images\\OCROutput\\OCRData.csv')
