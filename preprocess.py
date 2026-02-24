import json
import os
import glob
from tqdm import tqdm
from googletrans import Translator
import time

# ======================================================
# [설정] 데이터가 들어있는 최상위 폴더 경로
# ======================================================
ROOT_DIR = "/root/dataset/PoC_Dataset_v1/real" 
OUTPUT_FILE = "mdetr_train_data.json"

def convert_bbox_to_mdetr(bbox, img_w, img_h):
    """ [x, y, w, h] -> [cx, cy, w, h] (0~1 정규화) """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return [cx, cy, w, h]

def main():
    translator = Translator()
    all_data = []
    
    # JSON 파일 찾기
    json_files = glob.glob(os.path.join(ROOT_DIR, "**/*.json"), recursive=True)
    print(f"🚀 총 {len(json_files)}개의 JSON 파일을 찾았습니다. 변환 시작!")

    for file_path in tqdm(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # [수정됨] 이미지 경로 설정 (JSON 경로 -> PNG 경로로 변경)
            # 만약 이미지가 jpg라면 .replace('.json', '.jpg')로 바꿔주기
            image_path = file_path.replace('.json', '.png')
            
            # (선택) 실제로 이미지 파일이 있는지 확인하는 안전장치
            if not os.path.exists(image_path):
                # 경로가 안 맞으면 그냥 JSON 안의 파일명을 믿고 경로 조합 시도
                img_name_in_json = data['images']['file_name'] if isinstance(data['images'], dict) else data['images'][0]['file_name']
                parent_dir = os.path.dirname(file_path)
                image_path = os.path.join(parent_dir, img_name_in_json)

            # 이미지 정보
            img_info = data['images']
            if isinstance(img_info, list):
                img_w = img_info[0]['width']
                img_h = img_info[0]['height']
                img_id = img_info[0]['id']
            else:
                img_w = img_info['width']
                img_h = img_info['height']
                img_id = img_info['id']

            for ann in data['annotations']:
                korean_text = ann.get('referring_expression', "")
                
                # 번역 (API 오류 방지를 위한 예외처리 강화)
                english_text = ""
                if korean_text:
                    try:
                        translated = translator.translate(korean_text, src='ko', dest='en')
                        english_text = translated.text
                    except Exception:
                        # 번역 실패 시 대체 텍스트
                        cat_id = ann['category_id']
                        english_text = f"Find the object {cat_id}."
                        time.sleep(0.5) # API 차단 방지 딜레이

                normalized_bbox = convert_bbox_to_mdetr(ann['bbox'], img_w, img_h)

                sample = {
                    "image_id": img_id,
                    "file_name": image_path,  
                    "caption": english_text,
                    "bbox": normalized_bbox,
                    "label": ann['category_id']
                }
                all_data.append(sample)
                
        except Exception as e:
            print(f"⚠️ 에러 스킵 ({file_path}): {e}")
            continue

    # 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n🎉 완료! {OUTPUT_FILE} 생성됨. (데이터 {len(all_data)}개)")

if __name__ == "__main__":
    main()