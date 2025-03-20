# https://arxiv.org/pdf/2303.16755

template = {
    "template": '''
    Question: Which summary is the better one? An excellent summary is coherent, accurate, concise, and detailed. 
    Answer with A or B.

    Post:
    {}
        
    Summary A: {}

    Summary B: {}
    ''',
    "text_func": lambda text: text,
    "regex_pattern": r'(?|\*\*([AB])\*\*|([AB]))',
    "matched_char": 'A'
}
