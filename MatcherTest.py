import spacy
import re
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.language import Language

@Language.component("corporation_number_entity")
def corporation_number_entity(doc):
    # Get the current match and create tuple of entity label, start and end.
    # Append entity to the doc's entity. (Don't overwrite doc.ents!)
    #match_id, start, end = matches[i]
    #print(doc[end-1:end].text)

    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "corporate"}, {"LOWER": "number"}, {"IS_SPACE": True}, {"TEXT": {"REGEX": "[C][0-9]{7}"}}]
    matcher.add("corporate_number", [pattern])
    matches = matcher(doc)
    for match in matches:
        match_id, start, end = match
        entity = Span(doc, end-1, end, label="CORPORATE_NUMBER")
        doc.ents += (entity,)
    return doc
    
    #print(entity.text)

#reading the data
data = open('Sample.txt',encoding = 'cp850').read()

#if you get an error try the following
#data = open('11-0.txt',encoding = 'cp850').read()

# Import the Matcher
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
doc = nlp(data)
#nlp.add_pipe("corporation_number_pipe", after="ner")
#ruler = nlp.add_pipe("corporation_number_pipe", after="ner")
#patterns = [{"label": "corporation_number", "pattern": [{"LOWER": "corporate"}, {"LOWER": "number"}, {"IS_SPACE": True}, {"TEXT": {"REGEX": "[C][0-9]{7}"}}]}]
#ruler.add_patterns(patterns)

##print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents])
##print("="*80)
##matcher = Matcher(nlp.vocab)
##pattern = [{"TEXT": {"REGEX": "((?i)Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)"}}, {"ORTH": "-", "OP": "?"}, {"SHAPE": "dd"}]
##matcher.add("FILED_DATE", [pattern])
##matches = matcher(doc)
##doc = nlp(data)
##print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents])

#print("Matches:", [doc[start:end].text for match_id, start, end in matches])

expression = r"((?i)Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)[-]{0,1}[0-9]{2}[\s][0-9]{4}"
#expression = r"[C][0-9]{7}"
for match in re.finditer(expression, doc.text):
    start, end = match.span()
    span = doc.char_span(start, end)
    # This is a Span object or None if match doesn't map to valid token sequence
    if span is not None:
        print("Found match:", span.text)
        entity = Span(doc, start, end, label="FILED_DATE")
        doc.ents += (entity,)
print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents])
