DEBUG_MODE = False



class douban_dataset_config:
    data_path = './dataset/all.csv'
    train_path = './dataset/train.csv'
    test_path = './dataset/test.csv'


class model_config:
    bert_type = 'bert-base-chinese'
    fc_hidden_size = 500
    fc_num_layer = 3
    fc_dropout_rate = 0.5
    model_load_path = None
    bert_fine_tune = False