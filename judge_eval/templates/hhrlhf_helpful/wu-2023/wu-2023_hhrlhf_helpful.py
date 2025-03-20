# https://arxiv.org/pdf/2307.03025

template = {
    "template": '''
    [Question]
    {}
    [The Start of Assistant 1's Answer]
    {}
    [The Start of Assistant 2's Answer]
    {}
    [System]
    We would like to request your feedback on the performance of two AI assistants in response to the
    user question displayed above.
    Please rate the helpfulness, relevance, accuracy, level of details of their responses. First, provide
    your evaluation of the assistant's helpfulness, relevance, accuracy, and level of detail. Please
    provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring
    that the order in which the responses were presented does not affect your judgment.
    Once you have carefully reviewed both submissions, in a new line, choose between the answers of
    Assistant 1 and Assistant 2 by outputting the number 1 or 2 respectively. Do not output anything else other 
    than the number in this last line.
    ''',
    "text_func": lambda text: text,
    "regex_pattern": r'([1|2])$',
    "matched_char": '1'
}
