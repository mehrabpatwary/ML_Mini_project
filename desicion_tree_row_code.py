training_data = [
    ['Green', 3, 'Mango'],
    ['Yellow', 3, 'Mango'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]

# Column Labels
# These are used only print the tree
header = ['color', 'diameter', 'label']


def unique_vals(rows, col):
    # Find the unique values for a column in dataset
    return set([row[col] for row in rows])


###
# Demo:
# unique_vals(training_data, 0)
# unique_vals(training_data, 1)
###


def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]  # row = ['Green', 3, 'Mango'] # label = Mango
        if label not in counts:
            counts[label] = 0
        counts[label] += 1  # counts['Mango'] = 1 # {'Mango': 1}
    print(f'---------Step---------')
    # Return: {'Mango':2, 'Grape': 2, 'Lemon': 1}
    return counts  # ############# [5] --> gini


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:  # ############# [8]
    def __init__(self, column, value):
        self.column = column  # column = 0
        self.value = value  # value = 'Green'

    def match(self, example):  # ############# [13]
        # example:  ['Green', 3, 'Mango']
        val = example[self.column]  # column = 0 # val = Green
        if is_numeric(val):
            return val >= self.value
        else:
            # Return: True # Green == Green
            return val == self.value  # ############# [14] ---> partition [12]

    def __repr__(self):
        condition = '=='
        if is_numeric(self.value):  # value = 'Green'
            condition = '>='
        # Return: Is color == Green?
        return 'Is %s %s %s?' % (header[self.column],
                                 condition, str(self.value))
        # header = ['color', 'diameter', 'label']
        # ############# [9] --> find_best_split [7]


def partition(rows, question):  # ############# [11]
    # question:  Is color == Green?
    true_rows, false_rows = [], []
    for row in rows:
        # Row:  ['Green', 3, 'Mango']
        if question.match(row):  # ############# [12] ---> Question ---> match
            # True
            true_rows.append(row)  # ['Green', 3, 'Mango']
        else:
            # False
            false_rows.append(row)  # ['Yellow', 3, 'Mango']
    return true_rows, false_rows  # ############# [15] ---> find_best_split [10]


def gini(rows):
    count = class_counts(rows)  # ############# [4]
    # count_output: {'Mango':2, 'Grape': 2, 'Lemon': 1}
    impurity = 1
    for lbl in count:  # lbl: Mango, Grape, Lemon
        prob_of_lbl = count[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    # impurity_output: 0.6399999999999999
    return impurity  # ############# [6] ---> find_best_split [3]


def info_gain(left, right, current_uncertainty):  # ############# [17]
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
    # ############# [18] ---> gain


def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)  # ############# [3]
    # gini_output: 0.6399999999999999
    n_features = len(rows[0]) - 1
    # n_features: len = 2 - 1 = 1
    for col in range(n_features):  # Run one time # col = 0
        values = set([row[col] for row in rows])
        # values_Output: {'Green', 'Yellow', 'Red'}

        for val in values:
            # Column:  0 Value: Green
            question = Question(col, val)  # ############# [7]
            # Question_output: Is color == Green?
            true_rows, false_rows = partition(rows, question)  # ############# [10]
            # true_rows.append(row)  # ['Green', 3, 'Mango']
            # false_rows.append(row)  # ['Yellow', 3, 'Mango']
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)  # ############# [16]
            if gain >= best_gain:
                best_gain, best_question = gain, question
                # best_gain: 0.653 best_question: Is color == Green?
    return best_gain, best_question  # ############# [7] ---> build_tree [2]


class Leaf:
    def __init__(self, rows):
        self.prediction = class_counts(rows)


class Decision_node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_best_split(rows)  # ############# [2]
    # gain: 0.653 question: Is color == Green?
    if gain == 0:
        return Leaf(rows)  # ------> class_counts
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_node(question, true_branch, false_branch)


def print_tree(node, spacing=''):
    if isinstance(node, Leaf):
        print(spacing + 'Predict', node.prediction)
        return
    print(spacing + str(node.question))

    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + ' ')

    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + ' ')


def classify(row, node):
    if isinstance(node, Leaf):
        return node.prediction
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + '%'
    return probs


if __name__ == "__main__":
    my_tree = build_tree(training_data)  # ############# [1]
    print_tree(my_tree)

for row in training_data:
    print('Actual: %s. Prediction: %s' %
          (row[-1], print_leaf(classify(row, my_tree))))

