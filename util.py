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


