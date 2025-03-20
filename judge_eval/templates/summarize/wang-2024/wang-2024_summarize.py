# https://arxiv.org/pdf/2401.06080

template = {
    "template":
    '''
    As a neutral observer, your task is to assess the responses provided 
    by two TL;DR summarizations according to the same SUBREDDIT prompt shown below.
    Begin by comparing the two responses and provide a brief explanation. 
    Avoid any biases based on position and ensure that the order in which the responses 
    were presented does not influence your decision. Do not let the length of the responses influence your evaluation. 
    Do not favor certain names of the assistants. Strive to be as objective as possible.
    You need to choose only one of the two answers and respond by either A or B.
    
    {}
    
    A. {}

    B. {}
    
    Which one is better? A or B?
    ''',
    "text_func":
    lambda text: text + '.' if len(text) == 1 else text.split('\n')[0] + '\n' +
    text.split('\n')[-1],  #.strip().split('\n')[-1],
    "regex_pattern":
    r'([AB])[\.\s]',  #|([AB])\s*is better',
    "matched_char":
    'A'
}
