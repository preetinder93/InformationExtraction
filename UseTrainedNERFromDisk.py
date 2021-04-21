import spacy
from spacy.lang.en import English
import pandas as pd


#model = English().from_disk("C:\Python\Training\ML\InformationExtraction\TrainedModelLibrary\TrainedModelV1")
model = spacy.load("C:\\Python\\Training\\GIT\\InformationExtraction\\TrainedModelLibrary\\TrainedModelV10")
#model = spacy.load("C:\Python\Training\ML\InformationExtraction\TrainedModelLibrary\modelV2")
print(model.pipe_names)

text = 'State of California Secretary of State  Statement of Information FC02244 (Domestic Nonprofit, Credit Union and General Cooperative Corporations) Fy L E D  In the office of the Secretary of State of the State of California  Filing Fee: $20.00. If this is an amendment, see instructions. IMPORTANT â€” READ INSTRUCTIONS BEFORE COMPLETING THIS FORM     1. CORPORATE NAME WORLD MISSION FOR NEEDY OR ABUSED CHILDREN     FEB-17 2016 2. CALIFORNIA CORPORATE NUMBER C2225405 This Space for Filing Use Only Complete Principal Office Address (Do not abbreviate the name of the city. Item 3 cannot be a P.O. Box.)  3. STREET ADDRESS OF PRINCIPAL OFFICE IN CALIFORNIA, IF ANY STATE ZIP CODE 3008 HILLEGASS AVE, BERKELEY, CA 94705  4. MAILING ADDRESS OF THE CORPORATION STATE ZIP CODE TRACY JADE JORDAN 3008 HILLEGASS AVE, BERKELEY, CA 94705  Names and Complete Addresses of the Following Officers (The corporation must list these three officers. A comparable title for the specific officer may be added; however, the preprinted titles on this form must not be altered.)     5. CHIEF EXECUTIVE OFFICER/ ADDRESS CITY STATE ZIP CODE TRACY JADE JORDAN 3008 HILLEGASS AVE, BERKELEY, CA 94705  6. SECRETARY ADDRESS CITY STATE ZIP CODE CHITRANJAN SINGH 3008 HILLEGASS AVE, BERKELEY, CA 94705  7. CHIEF FINANCIAL OFFICER/ ADDRESS CITY STATE ZIP CODE  DENNIS CHAPMAN 3008 HILLEGASS AVE, BERKELEY, CA 94705  Agent for Service of Process [lf the agent is an individual, the agent must reside in California and Item 9 must be completed with a California street address, a P.O. Box address is not acceptable. If the agent is another corporation, the agent must have on file with the California Secretary of State a certificate pursuant to California Corporations Code section 1505 and Item 9 must be left blank.  8. NAME OF AGENT FOR SERVICE OF PROCESS  TRACY JADE JORDAN  9. STREET ADDRESS OF AGENT FOR SERVICE OF PROCESS IN CALIFORNIA, IF AN INDIVIDUAL CITY STATE ZIPCODE 3008 HILLEGASS AVE, BERKELEY, CA 94705  Common Interest Developments  10. [] Check here if the corporation is an association formed to manage a common interest development under the Davis-Stirling Common Interest Development Act, (California Civil Code section 4000, et seq.) or under the Commercial and Industrial Common Interest Development Act, (California Civil Code section 6500, et seq.). The corporation must file a Statement by Common Interest Development Association (Form SI-CID) as required by California Civil Code sections 5405(a) and 6760(a). Please see instructions on the reverse side of this form.     11. THE INFORMATION CONTAINED HEREIN IS TRUE AND CORRECT.  02/17/2016 TRACY JADE JORDAN     DATE TYPE/PRINT NAME OF PERSON COMPLETING FORM SIGNATURE  SI-100 (REV 01/2016) APPROVED BY SECRETARY OF STATE '
text = text.replace('\n', ' ').replace('\r', '').replace('\x0C','').replace('!@#$%^&*()[]{};:,./<>?\|`~-=_+', ' ').replace('\\','').replace('\"','')
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
