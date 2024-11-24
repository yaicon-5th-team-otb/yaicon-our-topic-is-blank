import random
from collections import defaultdict
import requests
import warnings
import urllib3
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
url = "https://absolute-hopeful-caiman.ngrok-free.app/api/generate"

# topics 불러오기
def load_topics_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content:
                print(f"파일 {file_path}가 비어 있습니다.")
                return []
            topics = json.loads(content)  # or json.load(file)
            return topics
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return []
    return topics

# 각 주제의 점수를 저장할 딕셔너리 (초기 점수는 1)
scores = defaultdict(lambda: 1)

# 비교 프롬프트 생성 함수
def generate_comparison_prompt(topic1, topic2, filename =r''):    # 프롬프트 파일 경로

    with open(filename, 'r', encoding='utf-8') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(topic1_title=topic1['title'], topic1_problem=topic1['problem'], topic1_motivation=topic1['motivation'],
        topic1_method=topic1['method'], topic1_plan="\n".join(topic1['plan']),
        topic2_title=topic2['title'], topic2_problem=topic2['problem'], topic2_motivation=topic2['motivation'],
        topic2_method=topic2['method'], topic2_plan="\n".join(topic2['plan']))
    
    return prompt

# 주제 비교 함수
def select_better_topic(topic1, topic2):
    # 주제 비교 프롬프트 생성
    prompt = generate_comparison_prompt(topic1, topic2)
    
    # API 요청 데이터 예시
    data = {
        "model": "0llama1:latest",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data, verify=False)
        if response.status_code == 200:
            result = response.json().get('response', '')
        else:
            print("에러:", response.status_code)
            print("에러 내용:", response.text)
            return None
    except requests.exceptions.RequestException as e:
        print("요청 중 오류 발생:", e)
        return None

    # 결과 분석 및 점수 반영
    answer = result.strip()[-1]
    if answer == "1":
        scores[topic1['title']] += 1
        return topic1
    elif answer == "2":
        scores[topic2['title']] += 1
        return topic2
    else:
        print(f"잘못된 형식의 응답: {result}")
        return None

# 단일 라운드 함수: 주제를 점수 순으로 정렬하여 쌍으로 비교
def single_round(topics):
    random.shuffle(topics)  # 첫 라운드에서는 주제를 무작위로 섞음
    topics.sort(key=lambda x: scores[x['title']], reverse=True)  # 점수 순으로 정렬
    
    next_round = []
    for i in range(0, len(topics) - 1, 2):
        better_topic = select_better_topic(topics[i], topics[i + 1])
        if better_topic:
            next_round.append(better_topic)
    
    # 주제 개수가 홀수인 경우 마지막 주제를 다음 라운드로 그대로 이동
    if len(topics) % 2 == 1:
        next_round.append(topics[-1])
    
    return next_round

# 최종 우승 주제를 선정하는 함수
def tournament_select_topic(topics, max_round=5):
    current_round = 0
    while len(topics) > 1 and current_round < max_round:
        print(f"\n라운드 {current_round + 1}: 남은 주제들: {[topic['title'] for topic in topics]}")
        topics = single_round(topics)
        current_round += 1
    
    # 최종 우승 주제를 점수에 따라 결정
    final_topic = max(topics, key=lambda x: scores[x['title']])
    return final_topic

file_path = r'' #deduplicated된 주제들 파일경로
topics = load_topics_from_json(file_path)

# 토너먼트 시작
print("***** Tournament Start *****")
final_topic = tournament_select_topic(topics)
print("\n***** Final Selected Topic *****")
print(f"Title: {final_topic['title']}")
print(f"Problem: {final_topic['problem']}")
print(f"Motivation: {final_topic['motivation']}")
print(f"Method: {final_topic['method']}")
print(f"Plan: {', '.join(final_topic['plan'])}")
