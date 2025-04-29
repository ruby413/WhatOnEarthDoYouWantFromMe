import re


# 1. 텍스트 정제 함수들

def remove_stopwords(tokens, stopword_list):
    stopword_set = stopword_list if isinstance(stopword_list, set) else set(stopword_list)
    return [token for token in tokens if token not in stopword_set]

def clean_text(text):
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_repetitions(text, repeat_limit=2):
    # 문자 반복 (예: ㅋㅋㅋㅋ → ㅋㅋ)
    text = re.sub(r'(.)\1{2,}', lambda m: m.group(1) * repeat_limit, text)

    # 음절 반복 (예: 하하하하 → 하하)
    text = re.sub(r'((..))\1{1,}', lambda m: m.group(1) * repeat_limit, text)

    return text


# 2. 텍스트 정제 + 토큰화 진행

def tokenize_and_clean_text(text, stopword_list=None, repeat_limit=2):
    text = normalize_repetitions(text, repeat_limit=repeat_limit)
    text = clean_text(text)
    tokens = text.split()
    if stopword_list:
        tokens = remove_stopwords(tokens, stopword_list)
    return tokens


# 3. 발화 단위 전처리

def preprocess_conversation_lines(
    text,
    stopwords=None,
    use_silence=False,
    speaker_token="[UTTER]",
    repeat_limit=2
):
    lines = text.strip().split('\n')
    results = []

    for line in lines:
        if not line.strip():
            processed = ["[SILENCE]"] if use_silence else []
        else:
            processed = tokenize_and_clean_text(line, stopword_list=stopwords, repeat_limit=repeat_limit)
            if not processed:
                processed = ["[SILENCE]"] if use_silence else []
            else:
                processed = [speaker_token] + processed

        results.append(processed)

    return results


# 4. flatten 함수

def flatten_utterances(utterance_tokens_list, sep_token=" "):
    return sep_token.join([" ".join(tokens) for tokens in utterance_tokens_list])


# 5.  전체 전처리 파이프라인

def preprocess(
    text,
    stopwords=None,
    speaker_token="[UTTER]",
    use_silence=True,
    sep_token=" ",
    repeat_limit=2  
):
    """
    전체 전처리 통합 함수
    """
    utterance_tokens = preprocess_conversation_lines(
        text,
        stopwords=stopwords,
        use_silence=use_silence,
        speaker_token=speaker_token,
        repeat_limit=repeat_limit
    )
    return flatten_utterances(utterance_tokens, sep_token=sep_token)
