# https://arxiv.org/pdf/2307.07889

template = {
    "template": '''
    Passage:
    {}

    Summary A: {}
    Summary B: {}

    Which Summary is more coherent, consistent, fluent and relevant relative to the passage, 
    Summary A or Summary B?
    ''',
    "text_func": lambda text: text,
    "regex_pattern": r'([AB])',
    "matched_char": 'A'
}
