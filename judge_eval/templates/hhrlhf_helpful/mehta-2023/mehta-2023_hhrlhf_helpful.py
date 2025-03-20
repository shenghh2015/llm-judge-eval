# https://arxiv.org/pdf/2312.00267

template = {
    "template": '''
    [System]
    Please act as an impartial judge and evaluate the quality of the
    responses provided by two AI assistants to the user question
    displayed below. You should choose the assistant that follows the
    user's instructions and answers the user's question better. Your
    evaluation should consider factors such as the helpfulness,
    relevance, accuracy, depth, creativity, and level of detail of
    their responses. Avoid any position biases and ensure that the
    order in which the responses were presented does not influence your
    decision. Do not allow the length of the responses to influence
    your evaluation. Do not favor certain names of the assistants. Be
    as objective as possible. Output your final verdict by strictly
    following this format: 'A' if assistant A is better, 'B' if
    assistant B is better. Output only that character and do not include 
    any other characters or spaces.
    [User Question]
    {}
    [The Start of Assistant A's Answer]
    {}
    [The End of Assistant A's Answer]
    [The Start of Assistant B's Answer]
    {}
    [The End of Assistant B's Answer]
    ''',
    "text_func": lambda text: text.replace("'", ''),
    "regex_pattern": r'([AB])',
    "matched_char": 'A'
}
