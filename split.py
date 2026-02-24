import json
import random
import os

# ======================================================
# [설정] 파일 경로 및 분할 비율 설정
# ======================================================
INPUT_FILE = "mdetr_train_data.json"  # 전처리 완료된 통합 파일
TRAIN_RATIO = 0.8  # 학습용 80%
VAL_RATIO = 0.1    # 검증용 10%

def save_json(data, filename):
    """ 데이터를 JSON 파일로 저장하는 함수 """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ 저장 완료: {filename:<15} (데이터 개수: {len(data)}개)")

def main():
    # 1. 통합 데이터 불러오기
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 오류: '{INPUT_FILE}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"📂 데이터를 로딩 중입니다: {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    total_count = len(all_data)
    print(f"📊 전체 데이터 개수: {total_count}개")

    # 2. 데이터 섞기 (Shuffle) - 재현성을 위해 시드 고정
    # 시드를 고정해야 나중에 다시 돌려도 똑같은 데이터끼리 묶인다
    random.seed(42)
    random.shuffle(all_data)

    # 3. 데이터 분할 (Slicing)
    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)
    
    # 리스트 슬라이싱으로 분할
    train_data = all_data[:train_count]
    val_data = all_data[train_count : train_count + val_count]
    test_data = all_data[train_count + val_count:]

    # 4. 결과 파일 저장
    print("\n🚀 데이터 분할 및 저장을 시작합니다...")
    save_json(train_data, "train.json")
    save_json(val_data, "val.json")
    save_json(test_data, "test.json")

    print("\n" + "="*40)
    print("🎉 [완료] 데이터셋 분할 작업이 끝났습니다!")
    print(f"   1. 학습용 (Train) : {len(train_data)}개")
    print(f"   2. 검증용 (Val)   : {len(val_data)}개")
    print(f"   3. 평가용 (Test)  : {len(test_data)}개")
    print("="*40)

if __name__ == "__main__":
    main()