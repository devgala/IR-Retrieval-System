from util import intersection
import json

def booleanRetrieval(query,corpus,redis):
    tokens = query.split(' ')
    return and_query(tokens,corpus,redis)

def and_query(query_terms,corpus,redis):
    # load postings in memory
    # postings, doc_freq = load_index_in_memory(corpus)

    # postings for only the query terms
    postings_for_keywords = {}
    doc_freq_for_keywords = {}

    for q in query_terms:
        if redis.exists(q)==1:
            # print(redis.lrange(q, 0, -1))
            postings_for_keywords[q] = redis.lrange(q, 0, -1)
        else:
            postings_for_keywords[q] = []

    # store doc frequency for query token in
    # dictionary

    for q in query_terms:
        if redis.exists(q)==1:
            doc_freq_for_keywords[q] =int(redis.hget('doc_freq',q))
        else:
            doc_freq_for_keywords[q] = 0

    # sort tokens in increasing order of their
    # frequencies

    sorted_tokens = sorted(doc_freq_for_keywords.items(), key=lambda x: x[1])
    # initialize result to postings list of the
    # token with minimum doc frequency

    result = postings_for_keywords[sorted_tokens[0][0]]
    # iterate over the remaining postings list and
    # intersect them with result, and updating it
    # in every step

    for i in range(1, len(postings_for_keywords)):
        result = intersection(result, postings_for_keywords[sorted_tokens[i][0]])
        if len(result) == 0:
            return result
    retreivedDocs  = []
    for docs in result:
        for obj in corpus['all_papers']:
            if docs==obj['docno']:
                retreivedDocs.append(obj)
                break
    print("No. of docs retrieved " + str(len(retreivedDocs)))            
    return json.dumps(retreivedDocs)
