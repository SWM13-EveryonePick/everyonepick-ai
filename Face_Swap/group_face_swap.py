from collections import defaultdict

''' 
사용자 별로 선호하는 사진 선택 결과 예시
id:A → group1, group2
id:B → group1
id:C → group3 
id:D → group1, group3
'''

group_photo_path = ["../img_data/group1.jpeg", "../img_data/group2.jpeg", "../img_data/group3.jpeg"]
user_choices = defaultdict(list)

user_choices['A'].extend([0, 1])
user_choices['B'].append(0)
user_choices['C'].append(2)
user_choices['D'].extend([0, 2])
