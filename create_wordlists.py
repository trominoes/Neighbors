from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import json
from names_dataset import NameDataset

# load pre-trained word embeddings model
print("Initializing... ")
# 400k-word GloVe model from https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html
glove_input_file = "neighbors/static/glove.6B.50d.txt"
model = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)

# load popular names dataset
nd = NameDataset()
first_names = nd.get_top_names(n=1000, use_first_names=True, country_alpha2='US')['US']
firstNamesList = first_names['M'] + first_names['F']
firstNamesList = [name.lower() for name in firstNamesList]
lastNamesList = nd.get_top_names(n=1000, use_first_names=False, country_alpha2='US')['US']
lastNamesList = [name.lower() for name in lastNamesList]

# load states and countries
states = [
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 
    'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 
    'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 
    'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 
    'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 
    'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 
    'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 
    'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 
    'washington', 'west virginia', 'wisconsin', 'wyoming'
]

single_word_countries = [
    'afghanistan', 'albania', 'algeria', 'andorra', 'angola', 'argentina', 
    'armenia', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain', 
    'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin', 
    'bhutan', 'bolivia', 'bosnia', 'botswana', 'brazil', 'brunei', 'bulgaria', 
    'burkina', 'burundi', 'cambodia', 'cameroon', 'canada', 'chile', 'china', 
    'colombia', 'comoros', 'congo', 'croatia', 'cuba', 'cyprus', 'denmark', 
    'djibouti', 'ecuador', 'egypt', 'eritrea', 'estonia', 'ethiopia', 'fiji', 
    'finland', 'france', 'gabon', 'gambia', 'georgia', 'germany', 'ghana', 
    'greece', 'grenada', 'guatemala', 'guinea', 'guyana', 'haiti', 'honduras', 
    'hungary', 'iceland', 'india', 'indonesia', 'iran', 'iraq', 'ireland', 
    'israel', 'italy', 'jamaica', 'japan', 'jordan', 'kenya', 'kiribati', 
    'kuwait', 'kyrgyzstan', 'laos', 'latvia', 'lebanon', 'lesotho', 'liberia', 
    'libya', 'liechtenstein', 'lithuania', 'luxembourg', 'madagascar', 'malawi', 
    'malaysia', 'maldives', 'mali', 'malta', 'mexico', 'micronesia', 'moldova', 
    'monaco', 'mongolia', 'morocco', 'mozambique', 'myanmar', 'namibia', 'nauru', 
    'nepal', 'nicaragua', 'niger', 'nigeria', 'norway', 'oman', 'pakistan', 
    'palau', 'panama', 'paraguay', 'peru', 'philippines', 'poland', 'portugal', 
    'qatar', 'romania', 'russia', 'rwanda', 'samoa', 'senegal', 'serbia', 'seychelles', 
    'singapore', 'slovakia', 'slovenia', 'somalia', 'spain', 'sudan', 'suriname', 
    'sweden', 'switzerland', 'syria', 'taiwan', 'tajikistan', 'tanzania', 'thailand', 
    'togo', 'tonga', 'tunisia', 'turkey', 'uganda', 'ukraine', 'uruguay', 'uzbekistan', 
    'vanuatu', 'venezuela', 'vietnam', 'yemen', 'zambia', 'zimbabwe'
]

# load blocked start words
blocklist = [
    "darfur", "ahmed", "malaysian", "melbourne", "chechnya", "sunni", "kurdish",
    "blacks", "scottish", "syrian", "norwegian", "swedish", "brazilian", "abdul",
    "ninth", "assault", "chirac", "racial", "nasdaq", "hezbollah", "iraqis",
    "militant", "abbas", "asean", "shiite", "putin", "milosevic", "russians",
    "philippine", "graf", "abdullah", "hispanic", "liverpool", "albanian",
    "puerto", "pyongyang", "qaeda", "tehran", "netanyahu", "nigerian", "mohammed",
    "kashmir", "indonesian", "orleans", "danish", "lincoln", "annan", "eighth", 
    "yugoslavia", "insurgents", "baltimore", "cuban", "hostage", "croatian",
    "yeltsin", "yugoslav", "hampshire", "tampa", "israelis", "females", "labour",
    "cleveland", "barack", "belgrade", "indians", "austrian", "ontario", 
    "lebanese", "nazi", "guerrillas", "musharraf", "jews", "francs", "forget",
    "metres"
]


def load_json(model):
    '''Creates json with valid words (keys) and 
    their neighbors (values) for use in Neighbors game
    Word2Vec model -> JSON written to static folder'''
    # selected words from the model must be reasonably common, valid
    selectedWords = model.index_to_key[2500:4500]
    cleaned_list = [word for word in selectedWords if is_valid(word)]
    np.random.shuffle(cleaned_list)

    print("Number of validated words: " + str(len(cleaned_list)))

    filename = "neighbors/static/wordlist.json"
    with open(filename, "w") as json_file:
    # write the dictionary to the file
        json.dump(cleaned_list, json_file)

    # adapt this length; can 'model' be truncated to just the top 10k words?
    # Get the first 10,000 words from the model
    top_10k_words = model.index_to_key[500:10000]

    # Extract the corresponding vectors
    top_10k_vectors = [model[word] for word in top_10k_words]

    # Create a new KeyedVectors object for the top 10,000 words
    top_10k_model = KeyedVectors(vector_size=model.vector_size)

    # Add the top 10k words and their vectors to the new KeyedVectors object
    top_10k_model.add_vectors(top_10k_words, top_10k_vectors)


    neighborsDict = {}

    # add 6 neighbors of each word to a dictionary
    for word in top_10k_words:
        result = top_10k_model.most_similar(positive=[word],topn=12)
        neighborsList = np.array(result)[:,0].tolist() # selects column of just word strings
        neighborsList = [neighbor for neighbor in neighborsList if is_valid_neighbor(word, neighbor)]
        neighborsDict[word] = neighborsList[0:6]

    # open a json file in write mode
    filename = "neighbors/static/neighbordict.json"
    with open(filename, "w") as json_file:
    # write the dictionary to the file
        json.dump(neighborsDict, json_file, indent=4)

    
def is_valid_neighbor(word, neighbor):
    if word in neighbor: 
        return False
    if neighbor in word: 
        return False
    for char in neighbor:
        if not char.isalpha():
            return False
    return True

def is_valid(word):
    '''Helper: checks validity of a word. All 
    characters must be alphabetic, and the word
    must be longer than two characters & not be a name/state.
    str word -> bool is_valid'''
    if len(word) <= 3:
        return False
    if word in firstNamesList or word in lastNamesList:
        return False
    if word in states or word in single_word_countries:
        return False
    if word in blocklist:
        return False
    for char in word:
        if not char.isalpha():
            return False
    return True


print("Generating jsons...")
load_json(model)

print("Complete!")