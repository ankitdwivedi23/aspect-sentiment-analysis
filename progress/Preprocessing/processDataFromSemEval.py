import sys
import xml.etree.ElementTree as ET

all_categories = ['ambience','anecdotes/miscellaneous','food','price','service']
all_sentiments = ['negative', 'positive', 'neutral', 'na']

def convertXmlToTsv(inputXmlFile, outputTsvFile):
    # sentiment label for each aspect
    sentimentLabels = {s[1] : s[0] for s in enumerate(all_sentiments)}
    
    tree = ET.parse(inputXmlFile)
    root = tree.getroot()
    
    with open(outputTsvFile, 'w+') as f:
        for sentence in root.iter('sentence'):
            fields = []
            fields.append(sentence.attrib['id'])
            fields.append(sentence.find('text').text)
            categories = []
            sentiments = []
            categorySentiments = dict()
            for aspectCategory in sentence.find('aspectCategories').iter('aspectCategory'):
                categories.append(aspectCategory.attrib['category'])
                sentiments.append(aspectCategory.attrib['polarity'])
                categorySentiments[aspectCategory.attrib['category']] = aspectCategory.attrib['polarity']
            
            fields.append(",".join(categories))

            # boolean label for each category
            fields.append(",".join(["1" if c in categories else "0" for c in all_categories]))
            fields.append(",".join(sentiments))

            fields.append(",".join([str(sentimentLabels[categorySentiments[c]]) if c in categories else str(sentimentLabels['na']) for c in all_categories]))
            f.write("\t".join(fields) + "\n")
    
if __name__ == '__main__':
    convertXmlToTsv(sys.argv[1], sys.argv[2])

