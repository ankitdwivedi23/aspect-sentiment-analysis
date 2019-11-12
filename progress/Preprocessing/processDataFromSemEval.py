import sys
import xml.etree.ElementTree as ET

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
            fields.append(",".join(polarities))
            f.write("\t".join(fields) + "\n")
    
if __name__ == '__main__':
    convertXmlToTsv(sys.argv[1], sys.argv[2])

