# This is a sample Python script.

import shutil
import os
import json
import numpy as np
import subprocess
import cProfile, pstats, io
from pstats import SortKey
import pandas as pd
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# reads s2 corpus in json and
# creates an intermediary file
# containing token and doc_id pairs.
    
class TrieNode:
    def __init__(self):
        self.children = {}  
        self.is_end_of_word = False  
        self.postings_list = []  

class Trie:
    def __init__(self):
        self.root = TrieNode()  

    def insert(self, word, doc_id):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()  
            node = node.children[char]
        node.is_end_of_word = True  
        if doc_id not in node.postings_list: 
            node.postings_list.append(doc_id) 

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return [] 
            node = node.children[char]
        if node.is_end_of_word:
            return node.postings_list 
        return []  
    
    def getNode(self,word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        if node.is_end_of_word:
            return node
        return node
    
    def getPostingList(self,node):
        if(node==None):
            return []
        if(node.is_end_of_word):
            return node.postings_list
        arr = []
        for c in node.children:
            arr.extend(self.getPostingList(node.children[c]))
        return arr
    
    def perfixSearch(self,word):
        node = self.getNode(word)
        return self.getPostingList(node)



def read_json_corpus(json_path):
    f = open(json_path + "/s2_doc.json", encoding="utf-8")
    json_file = json.load(f)
    if not os.path.exists(json_path + "/intermediate/"):
        os.mkdir(json_path + "/intermediate/")
    o = open(json_path + "/intermediate/output.tsv", "w", encoding="utf-8")
    for json_object in json_file['all_papers']:
        doc_no = json_object['docno']
        title = json_object['title'][0]
        paper_abstract = json_object['paperAbstract'][0]
        tokens = title.split(" ")
        for t in tokens:
            o.write(t.lower() + "\t" + str(doc_no) + "\n")
        tokens = paper_abstract.split(" ")
        for t in tokens:
            o.write(t.lower() + "\t" + str(doc_no) + "\n")
    o.close()


# sorts (token, doc_id) pairs
# by token first and then doc_id
def sort(dir):
    f = open(dir + "/intermediate/output.tsv", encoding="utf-8")
    o = open(dir + "/intermediate/output_sorted.tsv", "w", encoding="utf-8")

    # initialize an empty list of pairs of
    # tokens and their doc_ids
    pairs = []

    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        if len(split_line) == 2:
            pair = (split_line[0], split_line[1])
            pairs.append(pair)

    # sort (token, doc_id) pairs by token first and then doc_id
    sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))

    # write sorted pairs to file
    for sp in sorted_pairs:
        o.write(sp[0] + "\t" + sp[1] + "\n")
    o.close()


# converts (token, doc_id) pairs
# into a dictionary of tokens
# and an adjacency list of doc_id
def construct_postings(dir):
    # open file to write postings
    o1 = open(dir + "/intermediate/postings.tsv", "w", encoding="utf-8")

    postings = {}  # initialize our dictionary of terms
    doc_freq = {}  # document frequency for each term

    # read the file containing the sorted pairs
    f = open(dir + "/intermediate/output_sorted.tsv", encoding="utf-8")

    # initialize sorted pairs
    sorted_pairs = []

    # read sorted pairs
    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        pairs = (split_line[0], split_line[1])
        sorted_pairs.append(pairs)

    # construct postings from sorted pairs
    for pairs in sorted_pairs:
        if pairs[0] not in postings:
            postings[pairs[0]] = []
            postings[pairs[0]].append(pairs[1])
        else:
            len_postings = len(postings[pairs[0]])
            if len_postings >= 1:
                # check for duplicates
                # assuming the doc_ids are sorted
                # the same doc_ids will appear
                # one after another and detected by
                # checking the last element of the postings
                if pairs[1] != postings[pairs[0]][len_postings - 1]:
                    postings[pairs[0]].append(pairs[1])

    # update doc_freq which is the size of postings list
    for token in postings:
        doc_freq[token] = len(postings[token])

    # print("postings: " + str(postings))
    # print("doc freq: " + str(doc_freq))
    print("Dictionary size: " + str(len(postings)))

    # write postings and document frequency to file

    for token in postings:
        o1.write(token + "\t" + str(doc_freq[token]))
        for l in postings[token]:
            o1.write("\t" + l)
        o1.write("\n")
    o1.close()


# starting the indexing process
def index(dir):
    # reads the corpus and
    # creates an intermediary file
    # containing token and doc_id pairs.
    # read_corpus(dir)
    read_json_corpus(dir)

    # sorts (token, doc_id) pairs
    # by token first and then doc_id
    sort(dir)

    # converts (token, doc_id) pairs
    # into a dictionary of tokens
    # and an adjacency list of doc_id
    construct_postings(dir)

# return a dictionary from tsv file of postings
def load_index_in_memory(dir):
    f = open(dir + "intermediate/postings.tsv", encoding="utf-8")
    postings = {}
    doc_freq = {}

    for line in f:
        splitline = line.split("\t")

        token = splitline[0]
        freq = int(splitline[1])

        doc_freq[token] = freq

        item_list = []

        for item in range(2, len(splitline)):
            item_list.append(splitline[item].strip())
        postings[token] = item_list

    return postings, doc_freq

#intersection of sorted lists
def intersection(l1, l2):
    count1 = 0
    count2 = 0
    intersection_list = []

    while count1 < len(l1) and count2 < len(l2):
        if l1[count1] == l2[count2]:
            intersection_list.append(l1[count1])
            count1 = count1 + 1
            count2 = count2 + 1
        elif l1[count1] < l2[count2]:
            count1 = count1 + 1
        elif l1[count1] > l2[count2]:
            count2 = count2 + 1

    return intersection_list


def and_query(query_terms, corpus):
    # load postings in memory
    postings, doc_freq = load_index_in_memory(corpus)

    # postings for only the query terms
    postings_for_keywords = {}
    doc_freq_for_keywords = {}

    for q in query_terms:
        if q in postings:
            postings_for_keywords[q] = postings[q]
        else:
            postings_for_keywords[q] = []

    # store doc frequency for query token in
    # dictionary

    for q in query_terms:
        if q in doc_freq:
            doc_freq_for_keywords[q] = doc_freq[q]
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

    return result


def read_json_queries(json_path):
    f = open(json_path + "/s2_query.json", encoding="utf-8")
    json_file = json.load(f)
    query_dict = {}
    if not os.path.exists(json_path + "/intermediate/"):
        os.mkdir(json_path + "/intermediate/")
    for json_obj in json_file['queries']:
        qid = json_obj['qid']
        query = json_obj['query']
        query_dict[str(qid)] = []
        tokens = query.split(" ")
        for t in tokens:
            query_dict[str(qid)].append(t.lower())
        
    return query_dict

#boolean retrieval    
def retrieve_docs(query_dict,dir):
    o = open(dir + "/boolean_retrieval.tsv", "w", encoding="utf-8")
    for query in query_dict:
        result = and_query(query_dict[query],'s2/')
        ip = str(query) + '\t'
        for r in result:
            ip = ip + r + '\t'
        ip = ip + '\n'
        o.write(ip)
    o.close()

#create text file for grep
def create_txt(dir):
    f = open(dir + "/s2_doc.json", encoding="utf-8")
    json_file = json.load(f)
    o = open(dir + "/intermediate/grep_text.txt", "w", encoding="utf-8")
    for json_obj in json_file['all_papers']:
        doc_id = json_obj['docno']
        title = json_obj['title']
        abst = json_obj['paperAbstract']
        o.write(doc_id + ' ' + title[0] + ' ' + abst[0] + '\n')
    o.close()


def grep(query):
    return subprocess.run(['grep', query, 's2/intermediate/grep_text.txt'], capture_output=True, timeout=120)


def grep_retrieval(dir):
    create_txt(dir)
    query_dict = ["Fibered"]
    results = []
    for query in query_dict:
        pr = cProfile.Profile()
        # pr.enable()
        result = subprocess.run(['grep', query, 's2/intermediate/grep_text.txt'], capture_output=True, timeout=120)
        # pr.disable()
        results.append(result.stdout.decode())
        print(result.stdout.decode())
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
        

def boolean_retrieval(dir):
    query_dict = read_json_queries(dir)
    
    retrieve_docs(query_dict,dir)
        


def experiment_1(dir):

    #boolean retrival
    boolean_retrieval(dir)
    #using grep
    grep_retrieval(dir)


def preprocess_text(text):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower()) 
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos='v') for word in stemmed_tokens]  
    if not lemmatized_tokens:
        return None
    else:
        return ' '.join(lemmatized_tokens)

def experiment_2(dir):
    data = pd.read_csv(dir+'/intermediate/output.tsv', delimiter = '\t',header=None)
    custom_header = ['word', 'document']
    data.columns = custom_header
    data['word'] = data['word'].astype(str)
    data['word'] = data['word'].apply(preprocess_text)
    data = data.dropna()
    print(data.head())
    output_file_path = dir+'/intermediate/processed_output.tsv'
    data.to_csv(output_file_path, sep='\t', index=False, header= None)

def rotateList(arr):
    arr.append(arr[0])
    arr.pop(0)

def getRoatatedList(term):
    term = list(term)
    rotate = []
    i = len(term)-1
    while i >= 0:
        rotate.append(''.join(term))
        term.append(term[0])
        term.pop(0)
        i = i-1
    return rotate


def createPermutermIndex(dir):
    f = open(dir+'intermediate/postings.tsv', encoding="utf-8")
    o = open(dir+'intermediate/permuterm_index.tsv',"w", encoding="utf-8")

    for line in f:
        tokens = line.split("\t")
        term  = tokens[0]
        rotated_list = getRoatatedList(list(term+'$'))
        for permuterm in rotated_list:
            o.write(permuterm+"\t"+term+'\n')
    o.close()
    f.close()


def loadPermutermIndex(dir):
    f = open(dir+'intermediate/permuterm_index.tsv',encoding="utf-8")
    permutermDict = {}
    for line in f:
        token = line.split('\t')
        if token[0] not in permutermDict:
            permutermDict[token[0]] = []
            l = list(token[1])
        permutermDict[token[0]].append(''.join(l[:-1]))
        if len(permutermDict[token[0]]) > 1:
            print(token[0])
    return permutermDict

#for exp 4 and 5
def readWildCardQueries(dir):
    f = open(dir + "/s2_wildcard.json", encoding="utf-8")
    json_file = json.load(f)
    query_dict = {}
    if not os.path.exists(dir + "/intermediate/"):
        os.mkdir(dir + "/intermediate/")
    for json_obj in json_file['queries']:
        qid = json_obj['qid']
        query = json_obj['query']
        query_dict[str(qid)] = []
        tokens = query.split(" ")
        for t in tokens:
            query_dict[str(qid)].append(t.lower())
    return query_dict


def wildCardRetrievalLinear(dir):
    o = open(dir+'intermediate/wild_card_retrieval_linear.tsv',"w",encoding="utf-8")
    postings,freq = load_index_in_memory(dir)
    permuterm_index = loadPermutermIndex(dir)
    wild_card_queries = readWildCardQueries(dir)
    print(wild_card_queries)
    docs = {}
    for query in wild_card_queries:
        # print(wild_card_queries[query][0])
        cl = list(wild_card_queries[query][0]+'$')
        while cl[-1]!='*':
            rotateList(cl)
        cl = ''.join(cl[:-1])
        print(cl)
        words = set()
        for p in permuterm_index:
            if cl in p:
                for x in permuterm_index[p]:
                    words.add(x)
        print(words)
        for word in words:
            if query not in docs:
                docs[query] = []
            docs[query].extend(postings[word])

    s = ""
    for d in docs:
        s = str(d)
        for item in docs[d]:
            s = s + '\t' + item
        o.write(s+'\n')
    o.close()
    return docs

def createPermutermTrie(dir):
    f = open(dir+'intermediate/permuterm_index.tsv',encoding="utf-8")
    trie = Trie()
    for line in f:
        tokens = line.split('\t')
        l = list(tokens[1])
        trie.insert(tokens[0],''.join(l[:-1]))
    return trie

def wildCardRetrievalTrie(dir):
    o = open(dir+'intermediate/wild_card_retrieval_trie.tsv',"w",encoding="utf-8")
    
    trie_postings = createTriePostings(dir)
    permuterm_trie = createPermutermTrie(dir)
    queries = readWildCardQueries(dir)
    for q in queries:
        cl = list(queries[q][0]+'$')
        while cl[-1]!='*':
            rotateList(cl)
        cl = ''.join(cl[:-1])
        print(cl)
        words = permuterm_trie.perfixSearch(cl)
        print(words)
        for w in words:
            docs = trie_postings.search(w)
            s = str(q)
            for d in docs:
                s = s + '\t' + d
            s = s + '\n'
            o.write(s)
    o.close()
                


def experiment_4(dir):
    createPermutermIndex(dir)


    #part 1
    # wildCardRetrievalLinear(dir)
    wildCardRetrievalTrie(dir)
    #part 2

def createTriePostings(dir):
    data = pd.read_csv(dir+'/intermediate/output.tsv', delimiter = '\t',header=None)
    custom_header = ['word', 'document']
    data.columns = custom_header
    data['word'] = data['word'].astype(str)
    data['document'] = data['document'].astype(str)
    trie = Trie()
    for index, row in data.iterrows():        
        trie.insert(row['word'], row['document'])
    return trie

def experiment_3(dir):
    trie_postings = createTriePostings(dir)
    # print(trie_postings.search("correspondence").sort())
    queries = read_json_queries(dir)
    ans = []
    for i in queries:
        l = []
        for q in queries[i]:
            if not l.len():
                l = trie_postings.search(q)
            l = intersection(l, trie_postings(q))
        ans.append(l)
    return ans
        
    

# Code starts here
if __name__ == '__main__':
    #constructing inverted index
    index('s2/')
    # experiment_1('s2/')
    # experiment_2('s2/')
    # experiment_3('s2/')
    experiment_4('s2/')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
