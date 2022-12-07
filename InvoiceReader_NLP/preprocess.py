import spacy
from spacy.tokens import DocBin
import pickle
import re
nlp = spacy.blank("en")
# Load Data
training_data = pickle.load(open('./data/trainData.pickle','rb'))
testing_data = pickle.load(open('./data/testData.pickle','rb'))
print(training_data)
invalid_span_tokens = re.compile(r'\s')

# the DocBin will store the example documents
db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
#         valid_start = start
#         valid_end = end
#         while valid_start < len(text) and invalid_span_tokens.match(
#             text[valid_start]):
#             valid_start += 1
#         while valid_end > 1 and invalid_span_tokens.match(
#             text[valid_end - 1]):
#             valid_end -= 1
        span = doc.char_span(start, end, label=label)
        print(span)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./data/train.spacy")


# the DocBin will store the example documents
db_test = DocBin()
for text, annotations in testing_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        print(span)
        ents.append(span)
    doc.ents = ents
    db_test.add(doc)
db_test.to_disk("./data/test.spacy")
