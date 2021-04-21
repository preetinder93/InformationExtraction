from OCRImage import getOCROutput
import os
import pandas as pd

pdfDirectory = r'C:\Users\preet\workspace-neon\InformationExtraction\Images\SampleData\IE Training Data'
txtDirectory = r'C:\Users\preet\workspace-neon\InformationExtraction\Images\OCROutput\New'

def fetchOCRTxtFile():
    for filename in os.listdir(pdfDirectory):
        #print(filename)
        if filename.endswith(".pdf"):
            #print(os.path.join(pdfDirectory, filename))
            ocrResult = getOCROutput(os.path.join(pdfDirectory, filename), "string")
            
            if ocrResult is True:
                print(filename+": "+ocrResult)
                continue
            else:
                print("OCR suspended")
                break
        else:
            continue


def generateOCRDataCsv():
    ocrTextList = []
    for txtFilename in os.listdir(txtDirectory):
        if txtFilename.endswith(".txt"):
            print(txtFilename," read process starts")
            f = open(os.path.join(txtDirectory, txtFilename), "r")
            ocrText = f.read()
            f.close()
            print(txtFilename," read process ends")
            ocrTextList.append(ocrText)
    #print(ocrTextList)      
    df = pd.DataFrame(ocrTextList, columns = ['text'])
        
    df.to_csv("C:\\Users\\preet\\workspace-neon\\InformationExtraction\\Images\\OCROutput\\OCRData.csv", sep=',', index=False)
    print("CSV generated succesfully")

    
#fetchOCRTxtFile()
generateOCRDataCsv()
