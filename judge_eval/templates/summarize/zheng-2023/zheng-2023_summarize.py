# https://arxiv.org/pdf/2306.05685

template = {
    "template": '''
    [System]
    Please act as an impartial judge and evaluate the quality of two summaries to the post displayed below. \
    You should choose the summary that better summarizes the post. Your evaluation should consider factors such as \
    the helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin your evaluation by comparing the two summaries and providing a short explanation. \
    Avoid any position biases and ensure that the order in which the summaries were presented does not influence your decision. \
    Do not allow the length of the summaries to influence your evaluation. Do not favor certain names of the summaries. Be as objective as possible. \
    After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if Summary A is better, "[[B]]"if Summary B is better.

    [Post]
    {}

    [The Start of Summary A]
    {}
    [The End of Summary A]

    [The Start of Summary B]
    {}
    [The End of Summary B]
    ''',
    "text_func": lambda text: text,
    "regex_pattern": r'\[\[([AB])\]\]',
    "matched_char": 'A'
}
