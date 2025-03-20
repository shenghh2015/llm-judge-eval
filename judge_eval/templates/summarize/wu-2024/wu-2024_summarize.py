# https://arxiv.org/pdf/2307.03025

template = {
    "template": '''
    [Post]
    {}

    [The Start of Summary 1]
    {}
    [The End of Summary 1]

    [The Start of Summary 2]
    {}
    [The End of Summary 2]

    [System]
    We would like to request your feedback on the performance of two summaries on the same post displayed above.
    Please rate the helpfulness, relevance, accuracy, level of details of these summaries. \
    First, provide your evaluation of the summary's helpfulness, relevance, accuracy, and level of detail. \
    Please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that \
    the order in which the summaries were presented does not affect your judgment. Once you have carefully reviewed both summaries,\
    in a new line, choose between Summary 1 and Summary 2 by outputting the number 1 or 2 respectively. \
    Do not output anything else other than the number in this last line.
    ''',
    "text_func":
    lambda text: text.strip().split('\n')[-1],  # return the last line
    "regex_pattern": r'([12])\.?',
    "matched_char": '1'
}
