
def debug(raw_test_data, load_data, predictions):
    gt_data = load_data('gt.tsv', delimiter='\t', names=['label', 'message'], header=None)

    is_corrects = []
    for i, message in enumerate(raw_test_data):
        row = gt_data[gt_data['message'] == raw_test_data[i]]
        is_spam = 'spam' if predictions[i] == 1 else 'ham'

        if len(row) == 0:
            print(message)
            continue

        is_correct = row['label'].tolist()[0] == is_spam

        is_corrects.append(is_correct)

    print('P: {}%'.format(is_corrects.count(True) / len(is_corrects)))
