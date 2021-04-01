
import numpy as np

model = None
blocking_file = "..\data\X2_blocking_keys.csv"
clusters_file = "..\data\clustered_x2.csv"

class Record:
    """
    Data structure representing record titles.
    """
    def __init__(self, rid, rtitle, rblockingKey):
        self.id = rid
        self.title = rtitle
        self.blockingKey = rblockingKey

def read_file():
    """
    Read records from file preprocessed with blocking keys.
    """
    with open(blocking_file) as f:
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

# encoder-based similarity functions

def cosine_similarity(r1_encoded, r2_encoded):
    """
    Compute cosine similarity between two encoded vectors.
    """
    return np.dot(r1_encoded, r2_encoded) / (np.linalg.norm(r1_encoded) * np.linalg.norm(r2_encoded))

def calculate_encoding(r1, r2):
    """
    Encode row titles and calculate cosine distance between encodings.
    """
    if not model:
        model = prepare_encoder()

    # distance between the same row is 0
    if r1.id == r2.id:
        return 0

    r1_encoded = model([r1.title])[0]
    r2_encoded = model([r2.title])[0]
    sim = cosine(r1_encoded, r2_encoded)
    return sim

def prepare_encoder():
    """
    Prepare Universal Sentence Encoder model. Takes a few minutes to run.
    """
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print ("module %s loaded" % module_url)
    return model

def generate_similarity_table(distance_metric="jaccard"):
    """
    Generate table of similarities based on distance_metric.

       |  r1  |  r2  |  r3  |  r4  ...
       ----------------------------
    r1 |   0  |  0.7 |  0.3 |  0.4 ...
    r2 |  0.4 |   0  |  0.6 |  0.3 ...
    r3 |  0.5 |  0.2 |   0  |  0.1 ...
    r4 |  0.1 |  0.3 |  0.1 |   0  ...
    ...
    """
    records = read_file()
    distance_function = None
    distance_table = []
    titles = []
    if distance_metric == "jaccard":
        distance_function = calculate_jaccard
    elif distance_metric == "encoding":
        distance_function = calculate_encoding
    else:
        raise Exception('distance_metric should be either "jaccard" \
            or "encoding" to use the generate_similarity_table function.')

    for r1 in records:
        for r2 in records:
            distance_row = []
            distance = distance_function(r1, r2)
            distance_row.append(distance)
        distance_table.append(distance_row)
        titles.append(r1.rtitle)

    distance_df = pd.DataFrame(distance_table, columns=titles)
    distance_df.set_index(titles)
    return distance_df

# jaccard similarity functions

def find_clusters():
    """
    Identify clusters from the ground truth (Y) dataset.
    Used in run_jaccard_analysis function.
    """
    with open(clusters_file) as f:
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
    """
    Calculate jaccard similarity for 2 records. Return distance.
    """
    # if the records are the same, return 0
    if r1.id == r2.id:
        return 0

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
    """
    Identify titles for jaccard similarity (used in calculate_jaccard)
    """
    countIntersection = 0
    for token in title2:
        if token in title1:
            countIntersection += 1
    countUnion = len(title1) + len(title2) - countIntersection
    return countIntersection/countUnion


def tokenize(s):
    """
    Tokenize and preprocess strings
    """
    s = s.replace("/", " ")
    tokens = s.split()
    return tokens

def run_jaccard_analysis():
    """
    Create a jaccard.csv file by computing jaccard distance among rows inside a cluster
    and rows outside the cluster for analysis of jaccard metric.
    """
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
