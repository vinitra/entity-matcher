
import numpy as np

model = None

# encoder-based similarity functions

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def calculate_embeddings(u, v):
    if not model:
        model = prepare_encoder()
    u_vec = model([u])[0]
    v_vec = model([v])[0]
    sim = cosine(u_vec, v_vec)
    return sim

def prepare_encoder():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print ("module %s loaded" % module_url)
    return model

# jaccard similarity functions

class Record:
    def __init__(self, rid, rtitle, rblockingKey):
        self.id = rid
        self.title = rtitle
        self.blockingKey = rblockingKey

def read_file():
    with open("X2_blocking_keys_preprocessed.csv") as f:
        content = f.readlines()
        records = []
        firstLine = True
        for line in content:
            if (firstLine):
                firstLine = False
                continue
            attrs = line.split(",")
            id = attrs[1]
            title = attrs[14]
            #some times we have "\n" at the end of the blocking key
            blockingKey = attrs[17]
            blockingKey = blockingKey.split("\n")[0]
            rec = Record(id, title, blockingKey)
            #print(rec.id + rec.blockingKey)
            records.append(rec)
    return records

def find_clusters():
    with open("sigmod-contest-2021\data\clustered_x2.csv") as f:
        content = f.readlines()
        clusters = []
        firstLine = True
        cluster = set()
        for line in content:
            if (firstLine):
                firstLine = False
                continue

            attrs = line.split(",")
            if (attrs[0] == "-----"):
                clusters.append(cluster)
                cluster = set()
                continue
            cluster.add(attrs[0])
    return clusters


def calculate_jaccard(r1, r2):
    tokenizedBlKey1 = tokenize(r1.blockingKey)
    tokenizedBlKey2 = tokenize(r2.blockingKey)

    # if the records have different blocking key return 0
    if len(tokenizedBlKey1) != len(tokenizedBlKey2):
        return 0
    for token in tokenizedBlKey1:
        if not token in tokenizedBlKey2:
            return 0

    # otherwise tokenize titles and find their jaccard
    tokenizedTitle1 = tokenize(r1.title)
    tokenizedTitle2 = tokenize(r2.title)

    if (len(tokenizedTitle1) > len(tokenizedTitle2)):
        return calc_jac_titles(tokenizedTitle1, tokenizedTitle2)
    return calc_jac_titles(tokenizedTitle2, tokenizedTitle1)

#title1 has more tokens than 2
def calc_jac_titles(title1, title2):
    countIntersection = 0
    for token in title2:
        if token in title1:
            countIntersection += 1
    countUnion = len(title1) + len(title2) - countIntersection
    return countIntersection/countUnion


def tokenize(s):
    s = s.replace("/", " ")
    tokens = s.split()
    return tokens

def run_jaccard():
    records = read_file()
    clusters = find_clusters()
    resultString = "id,avgInnerSim,maxInnerSim,minInnerSim,avgOuterSim,maxOuterSim,minOuterSim\n"

    for r1 in records:
        cluster = set()
        maxInnerSim = 0
        sumInnerSim = 0
        minInnerSim = 1
        maxOuterSim = 0
        sumOuterSim = 0
        minOuterSim = 1

        # find the cluster of r1
        for c in clusters:
            if r1.id in c:
                cluster = c
                break

        for r2 in records:
            if r1.id != r2.id:
                jac = calculate_jaccard(r1, r2)
                if r2.id in cluster: # if records in the same cluster
                    # update inner similarities
                    sumInnerSim += jac
                    if (jac > maxInnerSim):
                        maxInnerSim = jac
                    if (jac < minInnerSim):
                        minInnerSim = jac
                else: # if records in different clusters
                    # update outer similarities
                    sumOuterSim += jac
                    if (jac > maxOuterSim):
                        maxOuterSim = jac
                    if (jac < minOuterSim):
                        minOuterSim = jac
        avgInnerSim = sumInnerSim/(len(cluster)-1)
        avgOuterSim = sumOuterSim/(len(records) - len(cluster) -1)
        resultString += (r1.id + "," + str(avgInnerSim) + "," + str(maxInnerSim)+ "," + str(minInnerSim) + "," + str(avgOuterSim) + "," + str(maxOuterSim) + "," + str(minOuterSim) + "\n")

    f = open("jaccard.csv", "w")
    f.write(resultString)
    f.close()
