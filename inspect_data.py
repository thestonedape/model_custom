import pickle

data = pickle.load(open('dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle', 'rb'))

print("="*80)
print("DATA STRUCTURE INSPECTION")
print("="*80)

print(f"\nTop level type: {type(data)}")
print(f"Top level keys (subjects): {list(data.keys())[:5]}")

subject = list(data.keys())[0]
print(f"\nSubject: {subject}")
print(f"Subject data type: {type(data[subject])}")
print(f"Number of sentences: {len(data[subject])}")

if isinstance(data[subject], list) and len(data[subject]) > 0:
    sentence = data[subject][0]
    print(f"\nFirst sentence type: {type(sentence)}")
    print(f"First sentence keys: {list(sentence.keys())}")
    
    if 'word' in sentence:
        print(f"\nNumber of words in first sentence: {len(sentence['word'])}")
        if len(sentence['word']) > 0:
            word = sentence['word'][0]
            print(f"First word type: {type(word)}")
            print(f"First word keys: {list(word.keys()) if isinstance(word, dict) else 'N/A'}")
            if isinstance(word, dict) and 'content' in word:
                print(f"First word content: {word['content']}")
