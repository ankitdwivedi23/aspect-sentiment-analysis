import sys
import xml.etree.ElementTree as ET

all_categories = ['ambience','anecdotes/miscellaneous','food','price','service']
all_polarities = ['negative', 'positive', 'neutral', 'conflict']

def convertXmlToTsv(inputXmlFile, outputTsvFile):
    tree = ET.parse(inputXmlFile)
    root = tree.getroot()
    
    with open(outputTsvFile, 'w+') as f:
        for sentence in root.iter('sentence'):
            fields = []
            fields.append(sentence.attrib['id'])
            fields.append(sentence.find('text').text)
            categories = []
            polarities = []
            for aspectCategory in sentence.find('aspectCategories').iter('aspectCategory'):
                categories.append(aspectCategory.attrib['category'])
                polarities.append(aspectCategory.attrib['polarity'])
            
            fields.append(",".join(categories))
            # boolean label for each category
            fields.append(",".join(["1" if c in categories else "0" for c in all_categories]))
            
            fields.append(",".join(polarities))
            # boolean label for each polarity
            fields.append(",".join(["1" if p in polarities else "0" for p in all_polarities]))
            f.write("\t".join(fields) + "\n")
    
if __name__ == '__main__':
    convertXmlToTsv(sys.argv[1], sys.argv[2])

