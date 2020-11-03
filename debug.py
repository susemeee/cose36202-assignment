
from sklearn.metrics import f1_score

def debug(raw_test_data, load_data, predictions):
    gt_data = load_data('gt.tsv', delimiter='\t', names=['label', 'message'], header=None)

    gts = []
    preds = []
    is_corrects = []

    for i, message in enumerate(raw_test_data):
        row = gt_data[gt_data['message'] == raw_test_data[i]]

        if len(row) == 0:
            print(message)
            continue

        is_spam = 1 if row['label'].tolist()[0] == 'spam' else 0
        is_correct = predictions[i] == is_spam

        is_corrects.append(is_correct)
        gts.append(is_spam)
        preds.append(predictions[i])

    print('P: {}%'.format(is_corrects.count(True) / len(is_corrects)))
    print('F1: {}%'.format(get_f1(gts, preds)))


def get_f1(gt, pred):
    score = f1_score(gt, pred)
    return score
