training_data = [
    ['Green', 3, 'Mango'],
    ['Yellow', 3, 'Mango'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]

All_Unique_Value = []

for i in range(len(training_data[0])):
    temp = (list(set([row[i] for row in training_data])))
    for x in temp:
        All_Unique_Value.append(x)

print(All_Unique_Value)

count = {}
for row in training_data:
    lbl = row[-1]
    if lbl not in count:
        count[lbl] = 0
    count[lbl] += 1
print(count)

num = 5.9
char = 'Red'
print(isinstance(num, int) or isinstance(num, float))
print(isinstance(char, int) or isinstance(char, float))
