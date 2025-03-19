PROMPT_TEMPLATES = {
    # summary task
    "summary":
    """
    Which of the following summaries does a better job of summarizing the most
    important points in the given forum post, without including unimportant or
    irrelevant details? A good summary is both precise and concise.
    
    Please answer the question in a json format, which contains two keys: "comparison" and "preferred".
    The "comparision" key should contain a one-sentence comparison of the two summaries, explaining which
    you prefer and why. 
    The "preferred" key should contain a binary value of "1" or "0", where "1" indicates that
    you prefer Summary A and "0" indicates that you don't prefer Summary A.
    
    The post and the two summaries are as follows:
    Post:
    {}
    Summary A: {}
    Summary B: {}
    """,
    # hhrlhf helpful task
    "hh_rlhf_helpful":
    """
    For the following query to a chatbot, which response  is more helpful?
    Please answer the question in a json format, which contains two keys: "comparison" and "preferred".
    The "comparision" key should contain a one-sentence comparison of the two responses, explaining which
    you prefer and why.
    The "preferred" key should contain a binary value of "1" or "0", where "1" indicates that
    you prefer Response A and believe it is more helpful and "0" indicates that you don't prefer Response A.
    The query and responses are shown below:
    Query: {}
    Response A: {}
    Response B: {}
    """,
    # general task
    "general":
    """
    Please answer the question in a json format, which contains two keys: "comparison" and "preferred".
    The "comparision" key should contain a one-sentence comparison of the two summaries, explaining which
    you prefer and why. 
    The "preferred" key should contain a binary value of "1" or "0", where "1" indicates that
    you prefer Response A and "0" indicates that you don't prefer Response A.
    Prompt: {}
    Reqsponse A: {}
    Response B: {}
    """
}
