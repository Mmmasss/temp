import torch
import torch.utils.data as tud
from torch.utils.data.sampler import Sampler
import torch.nn.utils.rnn as rnn_utils
from torch import nn
import numpy as np
import json
import math
import utils
import random
from pytorch_metric_learning import losses, distances, miners
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # 加载PCA算法包
timer = utils.Timer()


class MetricLearningDataset(tud.Dataset):
    def __init__(self, file_train, triplet_num, min_len, max_len, dataset_size=None, neg_rate=10):
        self.triplet_num = triplet_num
        self.min_len = min_len
        self.max_len = max_len
        self.dataset_size = dataset_size
        self.neg_rate = neg_rate
        self.trajs = None
        self.original_trajs = None
        self.dis_matrix = None
        self.sorted_index = None
        self.sim_matrix = None
        self.loss_map = None
        self.data_prepare(json.load(open(file_train)))

    def data_prepare(self, train_dict):
        """
        train_dict['trajs'] : list of list of idx
        train_dict['origin_trajs'] : list of list of (lon, lat)
        train_dict['dis_matrix'] : distance matrix
        """
        trajs = []
        original_trajs = []
        used_idxs = []
        x = []
        y = []
        for original_traj in train_dict["origin_trajs"]:
            x.extend([i[0] for i in original_traj])
            y.extend([i[1] for i in original_traj])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        self.meanx, self.meany, self.stdx, self.stdy = meanx, meany, stdx, stdy
        for idx, traj in enumerate(train_dict["trajs"]):
            if self.min_len < len(traj) < self.max_len:
                trajs.append(traj)
                original_traj = train_dict["origin_trajs"][idx]
                original_traj = [[(i[0] - meanx)/stdx, (i[1] - meany)/stdy] for i in original_traj]
                original_trajs.append(original_traj)
                used_idxs.append(idx)
        if self.dataset_size is None:
            self.dataset_size = len(used_idxs)
        else:
            self.dataset_size = min(self.dataset_size, len(used_idxs))
        used_idxs = used_idxs[:self.dataset_size]
        self.trajs = np.array(trajs[:self.dataset_size], dtype=object)
        self.original_trajs = np.array(original_trajs[:self.dataset_size], dtype=object)
        self.dis_matrix = np.array(train_dict["dis_matrix"])[used_idxs, :][:, used_idxs]
        self.sorted_index = np.argsort(self.dis_matrix, axis=1)
        a = 20
        self.sim_matrix = np.exp(-a * self.dis_matrix)
        self.loss_map = torch.zeros(self.dataset_size, self.dataset_size)
        for i in range(self.dataset_size):
            self.loss_map[i, i] = -1
            # self.loss_map[i, self.sorted_index[i, :self.triplet_num]] = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        anchor = self.trajs[idx]
        positive_idx = self.sorted_index[idx][:self.triplet_num+1].tolist()
        if idx in positive_idx:
            positive_idx.remove(idx)
        else:
            positive_idx = positive_idx[:self.triplet_num]
        positive = self.trajs[positive_idx]
        negative_idx = self.sorted_index[idx][-self.triplet_num*self.neg_rate:].tolist()
        list.reverse(negative_idx)
        negative_idx = np.random.choice(
            self.sorted_index[idx][self.triplet_num:],
            self.triplet_num * self.neg_rate).tolist()
        negative = self.trajs[negative_idx]
        trajs_a = self.original_trajs[idx]
        trajs_p = self.original_trajs[positive_idx]
        trajs_n = self.original_trajs[negative_idx]
        return anchor, positive, negative, trajs_a, trajs_p, trajs_n, idx, positive_idx, negative_idx, self.sim_matrix[idx, positive_idx], self.sim_matrix[idx, negative_idx], self.sim_matrix[idx, :]


def collate_fn(train_data):
    batch_size = len(train_data)
    anchor = [torch.tensor(traj[0]) for traj in train_data]
    anchor_lens = [len(traj) for traj in anchor]
    anchor = rnn_utils.pad_sequence(anchor, batch_first=True, padding_value=-1)

    pos = []
    for list_pos in [list(traj[1]) for traj in train_data]:
        pos.extend(list_pos)
    pos = [torch.tensor(pos_) for pos_ in pos]
    pos_lens = [len(pos_) for pos_ in pos]
    pos = rnn_utils.pad_sequence(pos, batch_first=True, padding_value=-1)

    neg = []
    for list_neg in [list(traj[2]) for traj in train_data]:
        neg.extend(list_neg)
    neg = [torch.tensor(neg_) for neg_ in neg]
    neg_lens = [len(neg_) for neg_ in neg]
    neg = rnn_utils.pad_sequence(neg, batch_first=True, padding_value=-1)

    trajs_a = [torch.tensor(np.array(traj[3]), dtype=torch.float32) for traj in train_data]
    trajs_a_lens = [traj.shape[0] for traj in trajs_a]
    trajs_a = rnn_utils.pad_sequence(trajs_a, batch_first=True, padding_value=0)

    trajs_p = []
    for list_trajs_p in [list(traj[4]) for traj in train_data]:
        trajs_p.extend(list_trajs_p)
    trajs_p = [torch.tensor(traj_p, dtype=torch.float32) for traj_p in trajs_p]
    trajs_p_lens = [traj.shape[0] for traj in trajs_p]
    trajs_p = rnn_utils.pad_sequence(trajs_p, batch_first=True, padding_value=0)

    trajs_n = []
    for list_trajs_n in [list(traj[5]) for traj in train_data]:
        trajs_n.extend(list_trajs_n)
    trajs_n = [torch.tensor(traj_n, dtype=torch.float32) for traj_n in trajs_n]
    trajs_n_lens = [traj.shape[0] for traj in trajs_n]
    trajs_n = rnn_utils.pad_sequence(trajs_n, batch_first=True, padding_value=0)

    anchor_idxs = torch.tensor([traj[6] for traj in train_data], dtype=torch.long)

    pos_idxs = []
    for list_pos_idx in [list(traj[7]) for traj in train_data]:
        pos_idxs.extend(list_pos_idx)
    pos_idxs = torch.tensor([pos_ for pos_ in pos_idxs])

    neg_idxs = []
    for list_neg_idx in [list(traj[8]) for traj in train_data]:
        neg_idxs.extend(list_neg_idx)
    neg_idxs = torch.tensor([neg_ for neg_ in neg_idxs])

    sim_pos = torch.tensor(np.array([traj[9] for traj in train_data]), dtype=torch.float32)
    sim_neg = torch.tensor(np.array([traj[10] for traj in train_data]), dtype=torch.float32)
    sim_matrix_a = torch.tensor(np.array([traj[11] for traj in train_data]), dtype=torch.float32)
    sim_matrix_a = sim_matrix_a[:, anchor_idxs]
    return anchor, anchor_lens, pos, pos_lens, neg, neg_lens, trajs_a,\
        trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,\
        anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a


class SeqHardSampler(Sampler):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        indices = []
        loss_map_copy = self.data.loss_map.clone()
        start_seq = [i for i in range(loss_map_copy.shape[0])]
        random.shuffle(start_seq)
        while start_seq:
            idx = start_seq.pop()
            indices.append(idx)
            if len(start_seq) < self.batch_size:
                indices.extend(start_seq)
                start_seq.clear()
                break
            losses, indexs = torch.topk(loss_map_copy[idx], k=self.batch_size-1, dim=0)
            loss_map_copy[:, indexs] = -1
            loss_map_copy[:, idx] = -1
            for index in indexs:
                start_seq.remove(index)
            indices.extend(indexs.tolist())
        indices = torch.LongTensor(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data)


class GlobalHardSampler(Sampler):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        indices = []
        datasize = self.data.dataset_size
        batch_num = math.ceil(datasize / self.batch_size)
        for i in range(batch_num):
            indices.append([])
        loss_map_copy = self.data.loss_map.clone()
        for i in range(datasize):
            torch.max(loss_map_copy)
        indices = torch.cat(indices, dim=0)
        return iter(indices)

    def __len__(self):
        return len(self.data)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.0, pe_max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(pe_max_len, emb_size)  # [pe_max_len, d_model]
        position = torch.arange(0, pe_max_len, dtype=torch.float).unsqueeze(1)  # [pe_max_len, 1]
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, max_len, d_model]
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = x.transpose(0, 1)
        return self.dropout(x)


class T3S(nn.Module):
    def __init__(self, vocab_size, emb_size, heads=8, encoder_layers=1, lstm_layers=1, pre_emb=None, t2g=None):
        super(T3S, self).__init__()
        self.lamb = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # lambda
        nn.init.constant_(self.lamb, 0.5)
        if pre_emb is not None:
            self.embedding = nn.Embedding(vocab_size, emb_size).from_pretrained(pre_emb)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.position_encoding = PositionalEncoding(emb_size, dropout=0.1)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            emb_size, heads, batch_first=True), encoder_layers)
        self.lstm = nn.LSTM(2, emb_size, lstm_layers, batch_first=True)
        self.aphla = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # alpha
        nn.init.constant_(self.aphla, 1)
        self.t2g = t2g
        self.mean_x = None
        self.mean_y = None
        self.std_x = None
        self.std_x = None

    def forward(self, x, trajs, trajs_lens):
        """
        x: LongTensor: [batch_size, seq_len]
        trajs: FloatTensor: [batch_size, seq_len, 2]
        trajs_lens: LongTensor: [batch_size]
        """
        # transformer embedding
        # seq_len = x.shape[1]
        mask_x = (x == -1)  # [batch_size, seq_len]
        # lens = seq_len - torch.sum(mask_x, dim=1)  # [batch_size]
        x = x.clamp_min(0)
        emb_x = self.embedding(x)  # emb_x: [batch_size, seq_len, dim_emb]
        emb_x = self.position_encoding(emb_x)
        # pe_emb_x[mask_x] = 0
        encoder_out = self.encoder(emb_x, src_key_padding_mask=mask_x)  # [batch_size, seq_len, dim_emb]
        # encoder_out[mask_x] = 0
        # encoder_out_mean = encoder_out.sum(dim=1) / lens.unsqueeze(1)  # [batch_size, dim_emb]
        encoder_out = torch.mean(encoder_out, 1)  # batch_size, dim_emb]
        # lstm embedding
        trajs = rnn_utils.pack_padded_sequence(trajs, trajs_lens, batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(trajs)
        # add
        output = self.lamb * encoder_out + (1 - self.lamb) * h_n.squeeze()  # output: [batch_size, dim_emb]
        return output

    def calculate_loss_vanilla(self, anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                               trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                               anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a, *args):
        batch_size = anchor.size(0)
        pos_num = pos.shape[0] // batch_size
        device = anchor.device
        output_a = self.forward(anchor, trajs_a, trajs_a_lens)
        output_p = self.forward(pos, trajs_p, trajs_p_lens)
        sim_p = torch.exp(-self.aphla * torch.norm(output_a.repeat(pos_num,
                          1) - output_p, dim=1)).reshape(batch_size, -1)
        sim_a = torch.exp(-self.aphla * torch.norm(output_a.unsqueeze(1)-output_a, dim=2)).reshape(batch_size, -1)
        w_p = torch.softmax(torch.ones(pos_num)/torch.arange(1, pos_num+1).float(), dim=0).to(device)
        loss_p = torch.sum(w_p * (sim_p - sim_pos)**2, dim=1)
        loss_n = torch.sum((torch.relu(sim_a - sim_matrix_a))**2, dim=1)
        loss = (loss_p + loss_n).mean()
        return loss, loss_p.mean(), loss_n.mean()

    def calculate_loss_vanilla_v2(self, anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                                  trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                                  anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a):
        batch_size = anchor.size(0)
        pos_num = pos.shape[0] // batch_size
        neg_num = neg.shape[0] // batch_size
        device = anchor.device
        output_a = self.forward(anchor, trajs_a, trajs_a_lens)
        output_p = self.forward(pos, trajs_p, trajs_p_lens)
        sim_p = torch.exp(-self.aphla * torch.norm(output_a.repeat(pos_num,
                          1) - output_p, dim=1)).reshape(batch_size, -1)
        output_n = self.forward(neg, trajs_n, trajs_n_lens)
        sim_n = torch.exp(-self.aphla * torch.norm(output_a.repeat(neg_num,
                          1) - output_n, dim=1)).reshape(batch_size, -1)
        w_p = torch.softmax(torch.ones(pos_num)/torch.arange(1, pos_num+1).float(), dim=0).to(device)
        w_n = torch.softmax(torch.ones(neg_num).float(), dim=0).to(device)
        loss_p = torch.sum(w_p * (sim_p - sim_pos)**2, dim=1)
        loss_n = torch.sum(w_n * (torch.relu(sim_n - sim_neg))**2, dim=1)
        loss = (loss_p + loss_n).mean()
        return loss

    def calculate_loss_seq_sampler(self, anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                                   trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                                   anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a, loss_map):
        batch_size = anchor.size(0)
        pos_num = pos.shape[0] // batch_size
        device = anchor.device
        output_a = self.forward(anchor, trajs_a, trajs_a_lens)
        output_p = self.forward(pos, trajs_p, trajs_p_lens)
        sim_p = torch.exp(-self.aphla * torch.norm(output_a.repeat(pos_num,
                          1) - output_p, dim=1)).reshape(batch_size, -1)
        sim_a = torch.exp(-self.aphla * torch.norm(output_a.unsqueeze(1)-output_a, dim=2)).reshape(batch_size, -1)
        w_p = torch.softmax(torch.ones(pos_num)/torch.arange(1, pos_num+1).float(), dim=0).to(device)
        loss_p = torch.sum(w_p * (sim_p - sim_pos)**2, dim=1)
        loss_n = torch.relu(sim_a - sim_matrix_a)
        loss_n_copy = loss_n.detach().cpu()
        idxs_diag = torch.arange(0,batch_size)
        loss_n_copy[idxs_diag,idxs_diag] = -1
        mask = loss_map[anchor_idxs]
        mask[:,anchor_idxs] = loss_n_copy
        loss_map[anchor_idxs] = mask
        loss_n = torch.sum(loss_n, dim=1)
        loss = loss_p.mean() + loss_n.mean()
        return loss, loss_p.mean(), loss_n.mean()

    def calculate_loss_hard_miner(self):
        pass

    def evaluate(self, test_loader, device, tri_num):
        self.eval()
        ranks = []
        hit_ratios_10 = []
        ratios10_50 = []
        pca_x = []
        with torch.no_grad():
            for (anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                 trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                 anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a) in test_loader:
                # 已经令dataloader的batch_size为len(dataset),所以这里一次性取出了所有的数据
                test_num = 200
                tb_anchor = anchor[:test_num].to(device)
                tb_trajs_a = trajs_a[:test_num].to(device)
                tb_trajs_a_lens = trajs_a_lens[:test_num]
                tb_pos_idxs = pos_idxs[:test_num*tri_num].reshape(test_num, tri_num).cpu().numpy()
                output_a = self.forward(tb_anchor, tb_trajs_a, tb_trajs_a_lens)
                # pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
                # pca_x = pca.fit_transform(output_a.cpu().numpy())  # 对样本进行降维
                bsz = 300
                sim_matrixs = []
                for i in range(len(anchor)//bsz+1):
                    lb = i * bsz
                    ub = min((i+1)*bsz, len(anchor))
                    output_b = self.forward(anchor[lb:ub].to(device), trajs_a[lb:ub].to(device), trajs_a_lens[lb:ub])
                    s = torch.exp(-torch.norm(output_a.unsqueeze(1) - output_b, dim=-1))
                    sim_matrixs.append(s)
                sim_matrix = torch.cat(sim_matrixs, dim=1).cpu().numpy()
                sorted_index = np.argsort(-sim_matrix, axis=1)
                sorted_index = sorted_index[:, 1:]
                for i in range(test_num):
                    avg_rank = 0
                    for idx in tb_pos_idxs[i]:
                        avg_rank += np.argwhere(sorted_index[i] == idx)[0][0]
                    avg_rank /= len(tb_pos_idxs[i])
                    hr_10 = len(np.intersect1d(tb_pos_idxs[i][:10], sorted_index[i][:10])) / 10 * 100
                    r10_50 = len(np.intersect1d(tb_pos_idxs[i][:10], sorted_index[i][:50])) / 10 * 100
                    ranks.append(avg_rank)
                    hit_ratios_10.append(hr_10)
                    ratios10_50.append(r10_50)
                break
        
        rank = np.mean(ranks)
        hr_10 = np.mean(hit_ratios_10)
        r10_50 = np.mean(ratios10_50)
        self.train()
        return rank, hr_10, r10_50, pca_x
