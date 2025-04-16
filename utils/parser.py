import argparse

def str_to_str_list(input_string):
    return [item for item in input_string.split(',')]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type = str, help = "specify model_path if you want to test a model")
    parser.add_argument('--dataset', default = 'TaoBao', type = str,
                        help = 'Dataset to use')

    parser.add_argument('--seed', default = 2022, type = int,
                        help = 'seed for experiment')
    parser.add_argument('--embed_size', default = 32, type = int,
                        help = 'embedding size for all layer')
    parser.add_argument('--lr', default = 0.05, type = float,
                        help = 'learning rate')
    parser.add_argument('--weight_decay', default = 8e-8, type = float,
                        help = "weight decay for adam optimizer")
    parser.add_argument('--model', default = 'dgrec', type = str,
                        help = 'model selection')
    parser.add_argument('--epoch', default = 1000, type = int,
                        help = 'epoch number')
    parser.add_argument('--patience', default = 10, type = int,
                        help = 'early_stop validation')
    parser.add_argument('--batch_size', default = 2048, type = int,
                        help = 'batch size')
    parser.add_argument('--layers', default = 1, type = int,
                        help = 'layer number')
    parser.add_argument('--gpu', default = 0, type = int,
                        help = '-1 for cpu, 0 for gpu:0')
    parser.add_argument('--k_list', default = [100, 300], type = list,
                        help = 'topk evaluation')
    parser.add_argument('--k', default = 20, type = int,
                        help = 'neighbor number in each GNN aggregation')
    parser.add_argument('--neg_number', default = 4, type = int,
                        help = 'negative sampler number for each positive pair')
    parser.add_argument('--metrics', type = str_to_str_list, default = ['recall', 'hit_ratio', 'coverage'], help="recall, hit_ratio, coverage, IUD, DILAD, ILAD, ILAD_DILAD") 
    parser.add_argument('--DCC_alpha', default = 0.1, type = float,
                        help = 'alpha for discounted coverage metric')
    parser.add_argument('--FDCC_alpha', default = 0.1, type = float,
                        help = 'alpha for frequency discounted coverage metric')
    parser.add_argument('--sigma', default = 1.0, type = float,
                        help = 'sigma for gaussian kernel')
    parser.add_argument('--gamma', default = 2.0, type = float,
                        help = 'gamma for gaussian kernel')
    parser.add_argument('--category_balance', default = True, type = bool,
                        help = 'whether make loss category balance')
    parser.add_argument('--beta_class', default = 0.9, type = float,
                        help = 'class re-balanced loss beta')
    parser.add_argument('--kernel',default='gaussian',type=str, 
                        help = 'kernel function for the similarity matrix: gaussian, poly, tanh')
    parser.add_argument('--submodular_selection_option',default='original',type=str, 
                        help = 'random bullshit we did to the submodular selection function: original, mean, sebastian')
    parser.add_argument('--popularity', default=False, type=bool, help = 'Popularity')

    args = parser.parse_args()
    return args

