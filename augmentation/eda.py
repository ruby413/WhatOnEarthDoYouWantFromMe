import pandas as pd
import random
import pickle
import re

# Load wordnet
with open("wordnet.pickle", "rb") as f:
    wordnet = pickle.load(f)

def get_only_hangul(line):
    return re.sub(r"[^ㄱ-ㅎ가-힣\s]", "", line)

def get_synonyms(word):
    synonyms = []
    try:
        for syn in wordnet.get(word, []):
            for s in syn:
                synonyms.append(s)
    except:
        pass
    return synonyms

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    return sentence.split(" ") if sentence else []

def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return new_words if new_words else [random.choice(words)]

def swap_word(new_words):
    if len(new_words) < 2:
        return new_words
    idx1, idx2 = random.sample(range(len(new_words)), 2)
    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def add_word(new_words):
    counter = 0
    while counter < 10:
        random_word = random.choice(new_words) if new_words else ""
        synonyms = get_synonyms(random_word)
        if synonyms:
            new_words.insert(random.randint(0, len(new_words)), synonyms[0])
            return
        counter += 1

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def EDA(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
    sentence = get_only_hangul(sentence)
    words = sentence.split()
    words = [word for word in words if word]
    num_words = len(words)
    if num_words == 0:
        return []

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    for _ in range(num_new_per_technique):
        augmented_sentences.append(' '.join(synonym_replacement(words, n_sr)))
        augmented_sentences.append(' '.join(random_insertion(words, n_ri)))
        augmented_sentences.append(' '.join(random_swap(words, n_rs)))
        augmented_sentences.append(' '.join(random_deletion(words, p_rd)))

    augmented_sentences = [get_only_hangul(s) for s in augmented_sentences]
    random.shuffle(augmented_sentences)
    return list(set(augmented_sentences[:num_aug])) + [' '.join(words)]