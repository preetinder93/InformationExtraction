import cv2
import pytesseract

def getOCROutput(image,outputFormat):
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
    if "string" in outputFormat:
            return pytesseract.image_to_string(img)
    elif "xml" in outputFormat:
            return pytesseract.image_to_alto_xml(img)


    ##cv2.imshow('Result', img)
    ##cv2.waitKey(0)




