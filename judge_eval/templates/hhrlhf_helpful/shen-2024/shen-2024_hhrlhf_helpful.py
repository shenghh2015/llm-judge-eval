# https://arxiv.org/pdf/2403.07708v2

template = {
    "template": '''
    Please act as an impartial judge and evaluate the quality of the responses
    provided by two AI assistants to the user question displayed below.  You should
    choose the assistant that follows the user's instructions better and provides
    more tailored responses to the user's questions.

    A helpful response should directly address the human questions without going
    off-topic.  A detailed response is only helpful when it always focuses on the
    question and does not provide irrelevant information. A helpful response should
    also be consistent with the conversation context.

    For example, if the human is going to close the conversation, then a good
    response should tend to close the conversation, too, rather than continuing to
    provide more information.  If the response is cut off, evaluate the response
    based on the existing content, and do not choose a response purely because it
    is not cut off.  Begin your evaluation by comparing the two responses and provide
    a short explanation.  Avoid any positional biases and ensure that the order in
    which the responses were presented does not influence your decision.  Do not
    allow the length of the responses to influence your evaluation.  Do not favor
    specific names of the assistants.
    
    Be as objective as possible.  After providing your explanation, output your final
    verdict by strictly following this format:  [[A]] if assistant A is better, [[B]]
    if assistant B is better. Please make sure the last word is your choice.
    --User Question-- 
    {}
    --The Start of Assistant A's Answer-- 
    {}
    --The End of Assistant A's Answer--
    --The Start of Assistant B's Answer--
    {}
    --The End of Assistant B's Answer--
    ''',
    "text_func": lambda text: text,
    "regex_pattern": r'\[\[([AB])\]\]',
    "matched_char": 'A'
}
