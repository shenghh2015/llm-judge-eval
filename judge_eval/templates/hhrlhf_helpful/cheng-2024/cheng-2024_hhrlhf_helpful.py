# https://arxiv.org/pdf/2311.08045

template = {
    "template": '''
    Here is a conversation:
    {}
    Here are the responses from two models [model_A], [model_B]:
    [model_A]: {}
    [model_B]: {}
    Please play the role of a judge, compare the responses of [model_A] and [model_B] in the above Q&A, and compare them
    according to the following standards, the importance of these standards decreases from front to back.
    - Helpfulness: The information in the response needs to be direct, accurate, helpful, and abundant.
    Please give the key reasons for the judgment from the above dimensions.
    Finally, in a new line, give the final answer from the following, not including other words:
    - [model_A] is better,
    - [model_B] is better.
    ''',
    "text_func": lambda text: text,
    "regex_pattern": r'\[model_([AB])\] is better',
    "matched_char": 'A'
}
