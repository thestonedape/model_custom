"""Debug dataset loading"""
import pickle
from data.vocabulary import Vocabulary

# Load vocabulary
vocab = Vocabulary(vocab_size=500)
vocab.load("data/vocabulary_top500.pkl")
print(f"Vocabulary size: {len(vocab.word2idx)}")
print(f"Sample words: {list(vocab.word2idx.keys())[:10]}")

# Load one pickle file
pickle_file = "dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle"
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

print(f"\nPickle file structure:")
print(f"  Type: {type(data)}")

if isinstance(data, dict):
    print(f"  Subjects: {list(data.keys())}")
    # Get first subject
    first_subject = list(data.keys())[0]
    sentences = data[first_subject]
    print(f"  Sentences for {first_subject}: {len(sentences)}")
    
    if sentences:
        first_sentence = sentences[0]
        print(f"\nFirst sentence structure:")
        print(f"  Keys: {first_sentence.keys() if isinstance(first_sentence, dict) else 'Not a dict!'}")
        
        if 'word' in first_sentence:
            words = first_sentence['word']
            print(f"  Number of words: {len(words)}")
            
            if words:
                first_word = words[0]
                print(f"\nFirst word structure:")
                print(f"  Keys: {first_word.keys() if isinstance(first_word, dict) else 'Not a dict!'}")
                
                if 'content' in first_word:
                    word_text = first_word['content'].lower()
                    print(f"  Word: '{word_text}'")
                    print(f"  In vocab: {vocab.is_in_vocabulary(word_text)}")
                
                # Check for EEG features
                print(f"\nAvailable EEG features:")
                for key in first_word.keys():
                    print(f"    {key}: {type(first_word[key])}")
