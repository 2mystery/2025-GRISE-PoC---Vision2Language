import json
import os

# 파일 경로 설정
INPUT_JSON = "/root/dataset/test.json"
OUTPUT_JSON = "/root/dataset/test_fixed.json"

print(f"📂 Loading {INPUT_JSON}...")
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ----------------------------------------------------
# [자동 감지 로직] 리스트인지 딕셔너리인지 확인
# ----------------------------------------------------
target_list = None

if isinstance(data, list):
    print("ℹ️ 감지된 구조: LIST 타입 ( [...] )")
    target_list = data
elif isinstance(data, dict):
    if 'images' in data:
        print("ℹ️ 감지된 구조: DICT 타입 ( { 'images': ... } )")
        target_list = data['images']
    else:
        print("⚠️ 'images' 키를 찾을 수 없는 딕셔너리 구조입니다.")
else:
    print("❌ 알 수 없는 JSON 구조입니다.")

# ----------------------------------------------------
# [경로 수정 로직] annotation -> rgb
# ----------------------------------------------------
count = 0
if target_list is not None:
    for item in target_list:
        # file_name 키가 있는지 확인
        if 'file_name' in item:
            original_path = item['file_name']
            
            # 경로 수정 조건
            if "/annotation/" in original_path:
                new_path = original_path.replace("/annotation/", "/rgb/")
                item['file_name'] = new_path
                count += 1

    print(f"✅ 총 {count}개의 이미지 경로를 '/annotation/' -> '/rgb/' 로 수정했습니다.")

    print(f"💾 Saving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print("🎉 완료! 이제 진짜 테스트 돌리러 가세요!")

else:
    print("❌ 수정할 데이터를 찾지 못했습니다. JSON 구조를 다시 확인해주세요.")