import sys
import xml.etree.ElementTree as ET
import collections

all_categories = ['ambience','miscellaneous','food','price','service']
all_sentiments = ['negative', 'positive', 'neutral', 'na']

category_mapping = {
    "RESTAURANT#GENERAL": "miscellaneous",
    "SERVICE#GENERAL": "service",
    "FOOD#QUALITY": "food",
    "FOOD#STYLE_OPTIONS": "food",
    "DRINKS#STYLE_OPTIONS": "food",
    "DRINKS#PRICES": "price",
    "RESTAURANT#PRICES": "price",
    "RESTAURANT#MISCELLANEOUS":"miscellaneous",
    "AMBIENCE#GENERAL": "ambience",
    "FOOD#PRICES": "price",
    "LOCATION#GENERAL": "miscellaneous",
    "DRINKS#QUALITY": "food",
    "FOOD#GENERAL": "food"
    }

def convertXmlToTsv(inputXmlFile, outputTsvFile):
    # sentiment label for each aspect
    sentimentLabels = {s[1] : s[0] for s in enumerate(all_sentiments)}
    
    tree = ET.parse(inputXmlFile)
    root = tree.getroot()
            
    with open(outputTsvFile, 'w+', encoding='utf-8') as f:
        for review in root.iter('Review'):
            for sentence in review.find('sentences').iter('sentence'):
                fields = []
                fields.append(sentence.attrib['id'])
                fields.append(sentence.find('text').text)
                opinions = sentence.find('Opinions')
                categories = []
                sentiments = []
                categorySentiments = dict()

                if opinions is None:
                    continue
                for aspectCategory in opinions.iter('Opinion'):
                    mappedAspectCategory = category_mapping[aspectCategory.attrib['category']]
                    if mappedAspectCategory not in categories:
                        categories.append(mappedAspectCategory)
                        sentiments.append(aspectCategory.attrib['polarity'])
                        categorySentiments[mappedAspectCategory] = aspectCategory.attrib['polarity']
            
                fields.append(",".join(categories))

                # boolean label for each category
                fields.append(",".join(["1" if c in categories else "0" for c in all_categories]))
                fields.append(",".join(sentiments))

                fields.append(",".join([str(sentimentLabels[categorySentiments[c]]) if c in categories else str(sentimentLabels['na']) for c in all_categories]))
                f.write("\t".join(fields) + "\n")       
    
if __name__ == '__main__':
    convertXmlToTsv(sys.argv[1], sys.argv[2])

