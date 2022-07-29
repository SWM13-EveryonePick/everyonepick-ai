from collections import defaultdict, Counter

''' 
사용자 별로 선호하는 사진 선택 결과 예시
id:A → group1, group2
id:B → group1
id:C → group3 
id:D → group1, group3
'''

group_photo_path = ["../img_data/group1.jpeg", "../img_data/group2.jpeg", "../img_data/group3.jpeg"]
user_choices = defaultdict(list)

# user_choices -> {'A': [0, 1], 'B': [0], 'C': [2], 'D': [0, 2]})
user_choices['A'].extend([0, 1])
user_choices['B'].append(0)
user_choices['C'].append(2)
user_choices['D'].extend([0, 2])

def find_base_photo(user_choices):
    # 아무도 선택 하지 않은 경우
    if len(user_choices) == 0:
        return -1

    result = []
    base_photo = []

    for choices in user_choices.values():
        result += choices

    # 사진 별 선택 수
    photo_choices = Counter(result).most_common()
    # base 사진, 받은 선택 수
    max_choice = photo_choices[0][1]

    # base 사진의 선택 수가 중복 되는지 확인
    for photo, choice in photo_choices:
        if choice == max_choice:
            base_photo.append(photo)

    return base_photo
