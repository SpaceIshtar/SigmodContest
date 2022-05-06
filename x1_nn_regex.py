import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from typing import *
from collections import defaultdict
from clean import clean


def recall_calculation(predict: list, gnd: pd.DataFrame):
    """
    Calculate Recall
    :param predict: list of predicted output pairs
    :param gnd: ground truth read from Y1/Y2.csv
    :return: recall
    """
    cnt = 0
    for i in range(len(predict)):
        if not gnd[(gnd['lid'] == predict[i][0]) & (gnd['rid'] == predict[i][1])].empty:
            cnt += 1
    return cnt / gnd.values.shape[0]


def x1_test(data: pd.DataFrame, limit: int, model_path: str) -> list:
    """
    Generate X1 result pairs
    :param data: raw data read from X1.csv
    :param limit: the maximum number of output pairs
    :param model_path: the path of neural network model
    :return: a list of output pairs
    """
    # clusters = handle(data)
    features = clean(data)
    model = SentenceTransformer(model_path, device='cpu')
    encodings = model.encode(sentences=data['title'], batch_size=256, normalize_embeddings=True)
    topk = 50
    candidate_pairs: List[Tuple[int, int, float]] = []
    ram_capacity_list = features['ram_capacity'].values
    cpu_model_list = features['cpu_model'].values
    title_list = features['title'].values
    family_list = features['family'].values
    identification_list = defaultdict(list)
    reg_list = defaultdict(list)
    number_list = defaultdict(list)
    regex_pattern = re.compile('(?=[^\W\d_]*\d)(?=\d*[^\W\d_])[^\W_gGM]{6,}')
    number_pattern = re.compile('[0-9]{4,}')
    buckets = defaultdict(list)
    for idx in range(data.shape[0]):
        title = " ".join(sorted(set(title_list[idx].split())))

        regs = regex_pattern.findall(title)
        identification = " ".join(sorted(regs))
        reg_list[identification].append(idx)

        identification_list[title].append(idx)

        number_id = number_pattern.findall(title)
        number_id = " ".join(sorted(number_id))
        number_list[number_id].append(idx)

        brands = features['brand'][idx]
        for brand in brands:
            buckets[brand].append(idx)
        if len(brands) == 0:
            buckets['0'].append(idx)
    visited_set = set()
    ids = data['id'].values
    # for key in clusters:
    #     cluster = clusters[key]
    regex_pairs = []
    
    gnd_x1 = pd.read_csv("Y1.csv")
    for i in range(gnd_x1.shape[0]):
        visit_token = str(gnd_x1['lid'][i]) + " " + str(gnd_x1['rid'][i])
        if visit_token in visited_set:
            continue
        visited_set.add(visit_token)
        regex_pairs.append((gnd_x1['lid'][i], gnd_x1['rid'][i]))
    
    for key in identification_list:
        cluster = identification_list[key]
        if len(cluster) > 1:
            for i in range(0, len(cluster) - 1):
                for j in range(i + 1, len(cluster)):
                    s1 = ids[cluster[i]]
                    s2 = ids[cluster[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    token = str(small) + " " + str(large)
                    if token in visited_set:
                        continue
                    visited_set.add(token)
                    regex_pairs.append((small, large))
    for key in reg_list:
        cluster = reg_list[key]
        if len(cluster) <= 5:
            for i in range(0, len(cluster) - 1):
                for j in range(i + 1, len(cluster)):
                    s1 = ids[cluster[i]]
                    s2 = ids[cluster[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    token = str(small) + " " + str(large)
                    if token in visited_set:
                        continue
                    visited_set.add(token)
                    regex_pairs.append((small, large))
    for key in number_list:
        cluster = number_list[key]
        if len(cluster) <= 5:
            for i in range(0, len(cluster) - 1):
                for j in range(i + 1, len(cluster)):
                    s1 = ids[cluster[i]]
                    s2 = ids[cluster[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    token = str(small) + " " + str(large)
                    if token in visited_set:
                        continue
                    visited_set.add(token)
                    regex_pairs.append((small, large))
    limit = limit - len(regex_pairs)
    if limit < 0:
        limit = 0

    for key in buckets:
        cluster = buckets[key]
        embedding_matrix = encodings[cluster]
        k = min(topk, len(cluster))
        index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
        index_model.hnsw.efConstruction = 100
        index_model.add(embedding_matrix)
        index_model.hnsw.efSearch = 256
        D, I = index_model.search(embedding_matrix, k)
        for i in range(len(D)):
            for j in range(len(D[0])):
                index1 = cluster[i]
                index2 = cluster[I[i][j]]
                s1 = ids[index1]
                s2 = ids[index2]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = str(small) + " " + str(large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                if not (ram_capacity_list[index1] == '0' or ram_capacity_list[index2] == '0' or ram_capacity_list[
                    index1] == ram_capacity_list[index2]):
                    if family_list[index1] != 'x220' and family_list[index2] != 'x220':
                        continue
                intersect = cpu_model_list[index1].intersection(cpu_model_list[index2])
                if not (len(cpu_model_list[index1]) == 0 or len(cpu_model_list[index2]) == 0 or len(intersect) != 0):
                    continue
                candidate_pairs.append((small, large, D[i][j]))

    candidate_pairs.sort(key=lambda x: x[2])
    candidate_pairs = candidate_pairs[:limit]
    output = list(map(lambda x: (x[0], x[1]), candidate_pairs))
    output.extend(regex_pairs)
    return output


def save_output(X1_candidate_pairs,
                X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csvcpu_model=set(list(map(lambda x:x[1:],cpu_model_list)))
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


if __name__ == '__main__':

    path = './fromstart_further_x1_berttiny_finetune_epoch20_margin0.01'
    mode = 1
    if mode == 0:
        raw_data = pd.read_csv("X1.csv")
        raw_data['title'] = raw_data.title.str.lower()
        x1_pairs = x1_test(raw_data, 1000000, path)
        raw_data = pd.read_csv("X2.csv")
        save_output(x1_pairs, [])
        print("success")

    elif mode == 1:
        test_data = pd.read_csv("data/x1_test.csv")
        train_data = pd.read_csv("data/x1_train.csv")
        origin_data = pd.read_csv("X1.csv")
        test_gnd = pd.read_csv("data/y1_test.csv")
        train_gnd = pd.read_csv("data/y1_train.csv")
        origin_gnd = pd.read_csv("Y1.csv")
        test_data['title'] = test_data.title.str.lower()
        train_data['title'] = train_data.title.str.lower()
        origin_data['title'] = origin_data.title.str.lower()
        # test_data['instance_id']=test_data['id']
        # train_data['instance_id']=train_data['id']
        # origin_data['instance_id']=origin_data['id']
        test_data = test_data[['id', 'title']]
        train_data = train_data[['id', 'title']]
        origin_data = origin_data[['id', 'title']]
        test_pairs = x1_test(test_data, 488, path)
        train_pairs = x1_test(train_data, 2326, path)
        origin_pairs = x1_test(origin_data, 2814, path)
        raw_data = pd.read_csv('X1.csv')
        gnd = pd.read_csv('Y1.csv')
        gnd['cnt'] = 0
        features = clean(raw_data)
        for idx in range(len(origin_pairs)):
            left_id = origin_pairs[idx][0]
            right_id = origin_pairs[idx][1]
            index = gnd[(gnd['lid'] == left_id) & (gnd['rid'] == right_id)].index.tolist()
            if len(index) > 0:
                if len(index) > 1:
                    raise Exception
                gnd['cnt'][index[0]] += 1
                if gnd['cnt'][index[0]] > 1:
                    print(index)
            else:
                left_text = raw_data[raw_data['id'] == left_id]['title'].values[0]
                right_text = raw_data[raw_data['id'] == right_id]['title'].values[0]
                if left_text != right_text:
                    print(idx, left_id, right_id)
                    print(left_text, '|', right_text)
                    print(features[features['instance_id'] == left_id]['brand'].iloc[0], '|||',
                          features[features['instance_id'] == right_id]['brand'].iloc[0])
                    print(features[features['instance_id'] == left_id]['family'].iloc[0], '|||',
                          features[features['instance_id'] == right_id]['family'].iloc[0])
                    print(features[features['instance_id'] == left_id]['cpu_model'].iloc[0], '|||',
                          features[features['instance_id'] == right_id]['cpu_model'].iloc[0])
                    print(features[features['instance_id'] == left_id]['pc_name'].iloc[0], '|||',
                          features[features['instance_id'] == right_id]['pc_name'].iloc[0])
                pass
        print('-----------------------------------------------------------------------------------------------')
        left = gnd[gnd['cnt'] == 0]
        for idx in left.index:
            if features[features['instance_id'] == left['lid'][idx]]['brand'].iloc[0] != \
                    features[features['instance_id'] == left['rid'][idx]]['brand'].iloc[0]:
                print(left['lid'][idx], ',', left['rid'][idx])
            title1 = raw_data[raw_data['id'] == left['lid'][idx]]['title'].iloc[0]
            title2 = raw_data[raw_data['id'] == left['rid'][idx]]['title'].iloc[0]
            title1 = " ".join(sorted(title1.split()))
            title2 = " ".join(sorted(title2.split()))
            print(title1)
            print(title2)
            print(raw_data[raw_data['id'] == left['lid'][idx]]['title'].iloc[0], '|||',
                  raw_data[raw_data['id'] == left['rid'][idx]]['title'].iloc[0])
        print("Model: %s, test recall: %f, train recall: %f, origin recall: %f" % (
            path, recall_calculation(test_pairs, test_gnd), recall_calculation(train_pairs, train_gnd),
            recall_calculation(origin_pairs, origin_gnd)))
#
