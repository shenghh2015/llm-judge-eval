# https://arxiv.org/pdf/2402.04792

template = {
    "template": '''
    A good summary is a shorter piece of text that has the essence of the original.
    It tries to accomplish the same purpose and conveys the key information from
    the original post.  Below we define four evaluation axes for summary quality:
    coherence, accuracy, coverage, and overall quality.
    Coherence:  This axis answers the question "how coherent is the summary on its
    own?" A summary is coherent if it's easy to understand when read on its own
    and free of English errors.  A summary is not coherent if it's difficult to
    understand what the summary is trying to say.  Generally, it's more important
    that the summary is understandable than it being free of grammar errors.
    Accuracy:  This axis answers the question "does the factual information in the
    summary accurately match the post?" A summary is accurate if it doesn't say
    things that aren't in the article, it doesn't mix up people, and generally is
    not misleading.
    Coverage:  This axis answers the question "how well does the summary cover the
    important information in the post?" A summary has good coverage if it mentions
    the main information from the post that's important to understand the situation
    described in the post.  A summary has poor coverage if someone reading only
    the summary would be missing several important pieces of information about
    the situation in the post.  A summary with good coverage should also match the
    purpose of the original post (e.g.  to ask for advice).
    Overall quality:  This axis answers the question "how good is the summary overall
    at representing the post?" This can encompass all of the above axes of quality,
    as well as others you feel are important.  If it's hard to find ways to make the
    summary better, the overall quality is good.  If there are lots of different ways
    the summary can be made better, the overall quality is bad.
    You are an expert summary rater.  Given a piece of text and two of its possible
    summaries, output 1 or 2 to indicate which summary best adheres to coherence,
    accuracy, coverage, and overall quality as defined above.
    
    Text - {}
    Summary 1 - {}
    Summary 2 - {}
    
    Preferred Summary=
    ''',
    "text_func": lambda text: text,
    "regex_pattern": r'([12])',
    "matched_char": '1'
}
