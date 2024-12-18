import random

seedArr = [1, 2, 3, 4, 5]
seed = 4

# seedArr 반복문 돌며 각 값을 seed로 설정
for s in seedArr:
    random.seed(s)  # seedArr의 값 사용
    arr = [1, 2, 3, 4, 5]
    random.shuffle(arr)
    print(f"Seed: {s}, Shuffled: {arr}")

# 한 번만 seed로 고정한 경우
random.seed(seed)  # 변수 seed 사용
arr = [1, 2, 3, 4, 5]
random.shuffle(arr)
print(f"Single Seed: {seed}, Shuffled: {arr}")
