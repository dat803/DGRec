import pdb
import logging
import pickle
import torch
import numpy as np
import math
from tqdm import tqdm
from scipy.stats import entropy
from torch.nn.functional import normalize
import os

from models.models import DGRec

class Tester(object):
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model
        # self.model_mf = args.model_mf
        self.history_dic = dataloader.historical_dict
        self.history_csr = dataloader.train_csr
        self.dataloader = dataloader.dataloader_test
        self.real_dataloader = dataloader
        self.test_dic = dataloader.test_dic
        self.cate = np.array(list(dataloader.category_dic.values()))
        self.metrics = args.metrics
        self.category_num = dataloader.category_num
        self.device = args.device

    def judge(self, users, items, k, **kwargs):
        results = {metric: 0.0 for metric in self.metrics}
        # for ground truth test
        # items = self.ground_truth_filter(users, items)
        both_ilad_dilad = "ILAD" in self.metrics and "DILAD" in self.metrics
        
        if "ILAD" in self.metrics or "DILAD" in self.metrics:
            embedding_provider = DGRec(self.args, self.real_dataloader)
            if not self.args.embedding_provider:
                print(f'ERROR: ILAD or DILAD selected, but no embedding_provider specified')
                exit(-1)
            embedding_provider_path = self.args.embedding_provider
            #logging.info(f'loading embedding provider from {embedding_provider_path}')
            embedding_provider.load_state_dict(torch.load(embedding_provider_path, map_location=self.args.device))   
            self.embedding_provider = embedding_provider.to(self.args.device)

        category_frequencies = self.category_frequencies(items)
        for metric in self.metrics:
            if metric == 'IUD' or (both_ilad_dilad and (metric == 'DILAD' or metric == 'ILAD')):
                continue

            f = Metrics.get_metrics(metric)
            for i in range(len(items)):
                results[metric] += f(items[i], test_pos_categories = [self.cate[j] for j in self.test_dic[users[i]]], test_pos = self.test_dic[users[i]], num_test_pos = len(self.test_dic[users[i]]), category_frequencies = category_frequencies[i][1], category_coverage = category_frequencies[i][0], model = self.model, k = k, category_num = self.category_num, DCC_alpha = self.args.DCC_alpha, FDCC_alpha = self.args.FDCC_alpha, DILAD_beta=self.args.DILAD_beta, device=self.args.device)

        if both_ilad_dilad:
            all_items_embs = self.embedding_provider.get_embedding()['item'][items]
            all_embeddings = normalize(all_items_embs, dim=2)

            all_distances = 1 - torch.einsum('uie, uje -> uij', all_embeddings, all_embeddings)

            beta = self.args.DILAD_beta

            if (not beta):
                raise Exception("Please specify DILAD_beta parameter.")

            num_users = len(users)

            weights = torch.full((num_users, k), beta, device=self.args.device)

            test_sets = [set(self.test_dic[user]) for user in users]

            for idx, (item_row, test_set) in enumerate(zip(items, test_sets)):
                mask = torch.tensor([1.0 if item.item() in test_set else beta for item in item_row], device=self.args.device)
                weights[idx] = mask

            embeddings_size_squared = k * (k - 1)

            ilad_all_users = torch.einsum('uij -> u', all_distances) / embeddings_size_squared
            dilad_all_users = torch.einsum('uij,ui,uj', all_distances, weights, weights) / embeddings_size_squared

            results["ILAD"] += torch.sum(ilad_all_users).item()
            results["DILAD"] += torch.sum(dilad_all_users).item()

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
        if not self.args.no_checkpoints:
            checkpoint_dir = self.args.output + "/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Try loading the latest checkpoint
        if not self.args.no_checkpoints and os.listdir(checkpoint_dir):
            latest_checkpoint = max([int(file.split('.')[0]) for file in os.listdir(checkpoint_dir) if file.endswith('.pkl')])
            start_batch = latest_checkpoint + 1

            with open(os.path.join(checkpoint_dir, f'{latest_checkpoint}.pkl'), 'rb') as f:
                checkpoint_data = pickle.load(f)
            results = checkpoint_data['results']
            count = checkpoint_data['count']
        else:
            start_batch = 0
            results = {}
            count = 0

            for k in self.args.k_list:
                results[k] = {metric: 0.0 for metric in self.metrics}

        h = self.model.get_embedding()
        all_recommended_items = torch.tensor(np.ndarray((0,300)))

        for batch_idx, batch in enumerate(tqdm(self.dataloader)):

            if batch_idx < start_batch:
                continue  # Skip already processed batches

            users = batch[0]
            count += users.shape[0]

            scores = self.model.get_score(h, users)

            users = users.tolist()
            mask = torch.tensor(self.history_csr[users].todense(), device = scores.device).bool()
            scores[mask] = -float('inf')

            _, recommended_items = torch.topk(scores, k=max(self.args.k_list))
            recommended_items = recommended_items.cpu()
            all_recommended_items = torch.cat([all_recommended_items, recommended_items])

            for k in self.args.k_list:
                results_batch = self.judge(users, recommended_items[:, :k], k=k, all_recommended_items = all_recommended_items)
                for metric in self.metrics:
                    results[k][metric] += results_batch[metric]

            if not self.args.no_checkpoints:
                # Save checkpoint
                checkpoint_data = {
                    'results': results,
                    'count': count,
                }
                with open(os.path.join(checkpoint_dir, f'{batch_idx}.pkl'), 'wb') as f:
                    pickle.dump(checkpoint_data, f)

        for k in self.args.k_list:
            for metric in self.metrics:
                if metric == "IUD":
                    nice = all_recommended_items[:,:k].to(self.device)
                    results[k][metric] = Metrics.IUD(nice, k)
                else:
                    results[k][metric] /= count

        self.show_results(results)

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
    def get_metrics(metric):
        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'coverage': Metrics.coverage,
            'DCC': Metrics.DCC,
            'FDCC': Metrics.FDCC,
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
    def DCC(items, **kwargs):
        alpha = kwargs['DCC_alpha']
        categories_covered = kwargs['category_coverage']
        category_num = kwargs['category_num']
        test_pos = kwargs['test_pos_categories']
        covered_truths = np.isin(categories_covered, test_pos).sum()
        dcc = 1 / category_num * (covered_truths + alpha * (len(categories_covered) - covered_truths))

        return dcc

    @staticmethod
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
    def IUD(items, k):
        IUD_batch_size = 10000
        all_items = items.unique()
        item_to_index = {item.item(): idx for idx, item in enumerate(all_items)}
        num_items = len(all_items)
        num_users = items.shape[0]
        IUD_batch_num = math.ceil(num_items/IUD_batch_size)
        iuds = 0

        item_matrix = torch.zeros((num_users, num_items), device=items.device)
        for i in range(num_users):
            item_indices = [item_to_index[item.item()] for item in items[i]]
            item_matrix[i, item_indices] = 1

        for i in range(IUD_batch_num):

            shared = item_matrix[(i*IUD_batch_size):((i+1)*IUD_batch_size),:] @ item_matrix.T

            difference_ratio = (k - shared) / k

            batch_iuds = difference_ratio.sum() / (num_users - 1)
            iuds+= batch_iuds

        return iuds/num_users