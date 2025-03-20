# https://arxiv.org/pdf/2305.18290

template = {
    "template": '''
    Which of the following summaries does a better job of summarizing the most
    important points in the given forum post, without including unimportant or
    irrelevant details? A good summary is both precise and concise. 

    Post:
    {}
        
    Summary A: {}

    Summary B: {}

    FIRST provide a one-sentence comparison of the two summaries, explaining which
    you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your
    choice. Your response should use the format: 
    Comparison: <one-sentence comparison and explanation> 
    Preferred: <"A" or "B">
    ''',
    "text_func": lambda text: text.replace('"', ''),
    "regex_pattern": r'Preferred:\s*([AB])',
    "matched_char": 'A'
}
