# https://arxiv.org/pdf/2304.00723

template = {
    "template": '''
    Question: Which summary is better-written and more consistent with the post? Please answer with one of the following options.
    
    The beginning of the post:
    {}
    
    Options:
    (A) {}
    (B) {}
    
    Answer: I will choose Option
    ''',
    "text_func": lambda text: text,

    # hard to capture all the cases
    # refer to stackoverflow answer:
    # https://stackoverflow.com/questions/28553012/grouping-and-or-in-regex
    "regex_pattern": r'(?|Option\s\(?([AB])\)?|\(([AB])\)|([AB])[\s,\.])',
    "matched_char": 'A'
}
