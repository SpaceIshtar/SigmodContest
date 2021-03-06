from FeatureExtracting import extract_x2
from EntityBlocking import block_x2
from x1_nn_regex import *


def save_output(pairs_x1, expected_size_x1, pairs_x2, expected_size_x2):
    if len(pairs_x1) > expected_size_x1:
        pairs_x1 = pairs_x1[:expected_size_x1]
    elif len(pairs_x1) < expected_size_x1:
        pairs_x1.extend([(0, 0)] * (expected_size_x1 - len(pairs_x1)))
    if len(pairs_x2) > expected_size_x2:
        pairs_x2 = pairs_x2[:expected_size_x2]
    elif len(pairs_x2) < expected_size_x2:
        pairs_x2.extend([(0, 0)] * (expected_size_x2 - len(pairs_x2)))
    output = pairs_x1 + pairs_x2
    output_df = pd.DataFrame(output, columns=['left_instance_id', 'right_instance_id'])
    output_df.to_csv('output.csv', index=False)


if __name__ == '__main__':
    path = './fromstart_further_x1_berttiny_finetune_epoch20_margin0.01'
    raw_data = pd.read_csv("X1.csv")
    raw_data['title'] = raw_data.title.str.lower()
    candidates_x1 = x1_test(raw_data, 1000000, path)
    raw_data = pd.read_csv('X2.csv')
    raw_data['name'] = raw_data.name.str.lower()
    features = extract_x2(raw_data)
    candidates_x2 = block_x2(features, 2000000)
    print('success')
    save_output(candidates_x1, 1000000, candidates_x2, 2000000)
