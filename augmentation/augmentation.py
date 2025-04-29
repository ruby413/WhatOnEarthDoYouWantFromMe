from .eda import EDA 
import pandas as pd

def augment_by_lines(text, num_aug=1, sep='\n', **kwargs):
    """
    
    """
    lines = text.strip().split('\n')
    augmented_lines = []
    for line in lines:
        if not line.strip():
            augmented_lines.append("")
            continue
        aug = EDA(line, num_aug=num_aug,**kwargs)
        augmented_lines.append(aug[0] if aug else line)
    return sep.join(augmented_lines)


def run_eda_augmentation(train_dataset, preprocess, **kwargs):
    """
    전체 대화 데이터를 줄 단위 증강하고 전처리한 후, 중복 제거까지 수행하는 파이프라인 함수.

    Args:
        train_dataset (pd.DataFrame): 원본 학습 데이터셋 ("conversation", "class" 컬럼 포함)
        preprocess (function): 전처리 함수

    Returns:
        pd.DataFrame: 전처리 + 증강 + 중복 제거된 결과 데이터프레임
    """
    augmented_data = []

    for idx, row in train_dataset.iterrows():
        conv = row.get("conversation", "")
        if not isinstance(conv, str) or not conv.strip():
            continue

        # 1. 원본 문장 (전처리 적용)
        original_text = preprocess(conv)
        augmented_data.append({
            "idx": idx,
            "conversation": original_text,
            "augmented": False,
            "class": row["class"]
        })

        # 2. 증강 문장 (줄 단위 증강 후 전처리 적용)
        aug_conv = augment_by_lines(conv, num_aug=1, **kwargs)
        aug_text = preprocess(aug_conv)
        augmented_data.append({
            "idx": idx,
            "conversation": aug_text,
            "augmented": True,
            "class": row["class"]
        })

    # 3. 데이터프레임 변환 및 중복 제거
    combined_df = pd.DataFrame(augmented_data)
    combined_df.sort_values(by="augmented", inplace=True)
    combined_df.drop_duplicates(subset=["conversation"], keep="first", inplace=True)

    # 4. 정렬 및 인덱스 재정렬
    combined_df.sort_values(by=["idx", "augmented"], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df
