# https://arxiv.org/pdf/2306.04181
template = {
    "template": '''
    You are a fair assessment expert, and you will be given one question along with 2 
    different responses. Your task is to decide which response is better. You should take 
    into consideration the accuracy, coherence, factuality, and comprehensiveness of 
    the responses to reach a judgment. Only return: “Response 1” or “Response 2”. 
    You do not need to explain the reason.
    Question: {} 
    Response 1: {} 
    Response 2: {}
    ''',
    "text_func": lambda text: text,
    "regex_pattern": r'Response ([1|2])',
    "matched_char": '1'
}
