import pdb
import logging
import torch
import numpy as np
import math
from tqdm import tqdm
from scipy.stats import entropy
from torch.nn.functional import normalize

class Tester(object):
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model
        # self.model_mf = args.model_mf
        self.history_dic = dataloader.historical_dict
        self.history_csr = dataloader.train_csr
        self.dataloader = dataloader.dataloader_test
        self.test_dic = dataloader.test_dic
        self.cate = np.array(list(dataloader.category_dic.values()))
        self.metrics = args.metrics
        self.category_num = dataloader.category_num
        self.runningIUD = False
        if self.runningIUD:
            self.runningIUD = True
            self.metrics.remove('IUD')

    def judge(self, users, items, **kwargs):
        results = {metric: 0.0 for metric in self.metrics}
        # for ground truth test
        # items = self.ground_truth_filter(users, items)
        category_frequencies = self.category_frequencies(items)
        for metric in self.metrics:
            if metric == 'IUD':
                continue

            f = Metrics.get_metrics(metric)
            for i in range(len(items)):
                item_cates = [(item, self.cate[item]) for item in items[i, :kwargs['k']]]
                test_pos_categories = [self.cate[item] for item in self.test_dic[users[i]]]
                good_alpha = Metrics.isAlphaGood(item_cates, kwargs['k'], category_frequencies[i], test_pos_categories)
                print(good_alpha)
                results[metric] += f(items[i], test_pos_categories = test_pos_categories, test_pos = self.test_dic[users[i]], num_test_pos = len(self.test_dic[users[i]]), category_frequencies = category_frequencies[i][1], category_coverage = category_frequencies[i][0], model = self.model, k = kwargs['k'], category_num = self.category_num, DCC_alpha = self.args.DCC_alpha, FDCC_alpha = self.args.FDCC_alpha)
        return results

    def ground_truth_filter(self, users, items):
        batch_size, k = items.shape
        res = []
        for i in range(len(users)):
            gt_number = len(self.test_dic[users[i]])
            if gt_number >= k:
                res.append(items[i])
            else:
                res.append(items[i][:gt_number])
        return res

    def test(self):
        results = {}
        if self.runningIUD:
            iud = {}
        h = self.model.get_embedding()
        count = 0

        for k in self.args.k_list:
            results[k] = {metric: 0.0 for metric in self.metrics}
            if self.runningIUD:
                iud[k] = 0

        for batch in tqdm(self.dataloader):

            users = batch[0]
            count += users.shape[0]
            # count += len(users)
            scores = self.model.get_score(h, users)

            # test ground truth
            # scores_ls = []
            # num_item = scores.shape[1]
            # for user in users:
            #     score_user = torch.zeros(num_item, device = scores.device)
            #     gt = torch.tensor(self.test_dic[user], device = scores.device)
            #     score_user[gt] = 1.0
            #     scores_ls.append(score_user)
            # scores = torch.stack(scores_ls)

            users = users.tolist()
            mask = torch.tensor(self.history_csr[users].todense(), device = scores.device).bool()
            scores[mask] = -float('inf')

            _, recommended_items = torch.topk(scores, k = max(self.args.k_list))
            recommended_items = recommended_items.cpu()
            for k in self.args.k_list:

                results_batch = self.judge(users, recommended_items[:, :k], k = k)

                for metric in self.metrics:
                    results[k][metric] += results_batch[metric]
                
                if self.runningIUD:
                    iud[k] += Metrics.IUD(recommended_items[:,:k], k = k)

        for k in self.args.k_list:
            for metric in self.metrics:
                results[k][metric] = results[k][metric] / count
            if self.runningIUD:
                iud[k] = iud[k]/count
            
        self.show_results(results)

        if self.runningIUD:
            for k in self.args.k_list:
                logging.info('for top {}, IUD = {}'.format(k, iud[k]))        

    def show_results(self, results):
        for metric in self.metrics:
            for k in self.args.k_list:
                logging.info('For top{}, metric {} = {}'.format(k, metric, results[k][metric]))
    
    def category_frequencies(self, items):
        recommended_categories = [np.unique(self.cate[item], return_counts=True) for item in items]
        return recommended_categories


class Metrics(object):

    def __init__(self):
        pass

    @staticmethod
    def choice(n, k):
        return math.factorial(n) / math.factorial(n - k)
    
    @staticmethod
    def precision(category_frequencies, ground_truth):
        truePositives = 0
        for i in range(len(category_frequencies[0])):
            if category_frequencies[0, i] in ground_truth:
                truePositives += category_frequencies[1, i]
        bottom = len(ground_truth) # maybe bottom should be the total length of recommended items instead
        precision = truePositives / bottom
        return precision
    
    @staticmethod
    def isAlphaGood(item_cates, k, category_frequencies, ground_truth_categories):
        #category_frequencies is a tuple list [(categories, times)] for a user
        #ground_truth_categories are the categories that are "correct"
        for i in range(1, k):
            p = Metrics.precision(category_frequencies, ground_truth_categories)
            choiceValue = Metrics.choice(i, k) * math.pow(p, i) * math.pow((1 - p), (k - i))
            for j in range(i - 1):
                
        return alpha 

    @staticmethod
    def get_metrics(metric):
        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'coverage': Metrics.coverage,
            'DCC': Metrics.DCC,
            'FDCC': Metrics.FDCC,
            'ILAD': Metrics.ILAD,
            'DILAD': Metrics.DILAD,
        }

        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/num_test_pos

    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def coverage(items, **kwargs):

        count = kwargs['category_coverage']

        return count.size
    
    @staticmethod
    # Discounted Category Coverage
    def DCC(items, **kwargs):
        alpha = kwargs['DCC_alpha']
        categories_covered = kwargs['category_coverage']
        category_num = kwargs['category_num']
        test_pos = kwargs['test_pos_categories']
        covered_truths = np.isin(categories_covered, test_pos).sum()
        dcc = 1 / category_num * (covered_truths + alpha * (len(categories_covered) - covered_truths))

        return dcc
    
    @staticmethod
    # Frequency aware Discounted Category Coverage
    def FDCC(items, **kwargs):
        fdcc = 0
        alpha = kwargs['FDCC_alpha']
        category_num = kwargs['category_num'] #C_I
        categories_covered = kwargs['category_coverage'] #C_R_u
        ground_truth = np.unique(kwargs['test_pos_categories'], return_counts=True) # = [[categories],[how many times each category appears]]

        intersect = np.intersect1d(categories_covered, ground_truth[0])

        for c in intersect:
            F_g_u = ground_truth[1][list(ground_truth[0]).index(c)] # we just get the amount of times category c appears in the ground truth
            base = 2
            if (F_g_u < base):
                fdcc += 1
            else:
                fdcc += math.log(F_g_u, base)
        
        fdcc += alpha * (len(categories_covered) - len(intersect))
        fdcc /= category_num
        return fdcc
    
    @staticmethod
    def IUD (items, **kwargs):
        iuds = 0
        k = kwargs['k']
        number_users = items.shape[0]
        for i in range(number_users):
            same_recommendations = torch.isin(items, items[i]).sum(dim=1)
            different_recommendations = torch.ones(number_users)*k-same_recommendations
            difference_ratio = torch.div(different_recommendations,k)
            iuds += (1/(number_users-1))*torch.sum(difference_ratio)
        return iuds.item()
    
    @staticmethod
    def ILAD(items, model, **kwargs):
        embeddings = normalize(model.get_embedding()['item'][items])

        distance = 1 - torch.mm(embeddings, embeddings.t())

        result = torch.sum(distance) / (embeddings.size()[0]**2)
        return result
    
    @staticmethod
    def DILAD(items, model, test_pos, **kwargs):
        embeddings = normalize(model.get_embedding()['item'][items])

        distance = 1 - torch.mm(embeddings, embeddings.t())

        beta = 0.1
        weights = torch.full((1, len(items)), beta, device='cuda')

        for idx, item in enumerate(items):
            if item in test_pos:
                weights[0][idx] = 1

        result = (1 / (len(embeddings) ** 2)) * torch.mm(torch.mm(weights, distance), weights.T)
        
        return result.item()
