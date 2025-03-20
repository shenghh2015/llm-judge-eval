# https://arxiv.org/pdf/2305.18201
template = {
    "template": '''
    QUESTION: {}
    ANSWER1: {}
    ANSWER2: {}
    TASK: Choose the better answer.
    BETTER ANSWER: ANSWER1 (or ANSWER2) is better.
    ''',
    "text_func": lambda text: text,
    "regex_pattern": r'ANSWER([1|2]) is better.',
    "matched_char": '1'
}
