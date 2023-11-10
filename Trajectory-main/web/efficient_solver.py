import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import math
import sys

sys.path.append('../')


class EfficientSolver:
    def __init__(self, model_path, device, mean_x, mean_y, std_x, std_y):
        self.model = torch.load(model_path)
        self.model.to(device)
        self.model.eval()
        self.t2g = self.model.t2g
        self.device = device
        self.max_bsz = 100
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.std_x = std_x
        self.std_y = std_y
    
    def normalize_traj(self, traj):
        """
        归一化轨迹
        :param traj:
        :return: normalized_traj: List
        """
        return [((i[0] - self.mean_x) / self.std_x, (i[1] - self.mean_y) / self.std_y) for i in traj]

    def compute_similarity(self, query_embeddings, target_embeddings) -> np.array:
        """
        计算轨迹之间的相似度
        :param query_embeddings: Array [bsz1, embedding_size]
        :param target_embeddings: Array [bsz2, embedding_size]
        :return:similarity matrix: Array [bsz1, bsz2]
        """
        bsz1 = query_embeddings.shape[0]
        bsz2 = target_embeddings.shape[0]
        similarity = []
        for i in range(math.ceil(bsz1 / self.max_bsz)):
            lb1 = i * self.max_bsz
            ub1 = min((i + 1) * self.max_bsz, bsz1)
            b_q_embeddings = torch.tensor(query_embeddings[lb1:ub1], device=self.device).unsqueeze(1)
            b_sims = []
            for j in range(math.ceil(bsz2 / self.max_bsz)):
                lb2 = j * self.max_bsz
                ub2 = min((j + 1) * self.max_bsz, bsz2)
                b_t_embeddings = torch.tensor(target_embeddings[lb2:ub2], device=self.device)
                b_sim = torch.exp(-torch.norm(b_q_embeddings - b_t_embeddings, dim=-1))
                b_sims.append(b_sim.cpu().numpy())
            b_sims = np.concatenate(b_sims, axis=1)
            similarity.append(b_sims)
        similarity = np.concatenate(similarity, axis=0)
        return similarity

    def embed_trajectory(self, trajectory):
        """
        生成单个轨迹的embedding
        :param trajectory: 经纬度坐标轨迹
        :return:embedding: np.array
        """
        traj_1d, coord_traj = self.t2g.convert1d(trajectory)
        traj_1d = torch.tensor(traj_1d, dtype=torch.long).unsqueeze(0).to(self.device)
        traj = torch.tensor(self.normalize_traj(coord_traj), dtype=torch.float32).unsqueeze(0).to(self.device)
        traj_lens = torch.tensor([len(traj)], dtype=torch.long)
        embedding = self.model.forward(traj_1d, traj, traj_lens)
        return embedding.detach().cpu().numpy()[0]

    def embed_trajectory_batch(self, trajectories):
        """
        批量生成embedding
        :param trajectories: 经纬度坐标轨迹
        :return:embedding: List[np.array]
        """
        traj_1ds = []
        coord_trajs = []
        for trajectory in trajectories:
            traj_1d, coord_traj = self.t2g.convert1d(trajectory)
            traj_1ds.append(traj_1d)
            coord_trajs.append(coord_traj)
        traj_1d = [torch.tensor(traj_1d, dtype=torch.long) for traj_1d in traj_1ds]
        normalized_traj = [torch.tensor(self.normalize_traj(t), dtype=torch.float32) for t in coord_trajs]
        traj_lens = torch.tensor([traj.shape[0] for traj in normalized_traj], dtype=torch.long)
        traj_1d = rnn_utils.pad_sequence(traj_1d, batch_first=True, padding_value=-1)
        normalized_traj = rnn_utils.pad_sequence(normalized_traj, batch_first=True, padding_value=0)
        # 太大会把内存搞爆,分批次处理

        embeddings = []
        for i in range(math.ceil(traj_1d.shape[0] / self.max_bsz)):
            lb = i * self.max_bsz
            ub = min((i + 1) * self.max_bsz, traj_1d.shape[0])
            batch_traj_1d = traj_1d[lb:ub].to(self.device)
            batch_traj = normalized_traj[lb:ub].to(self.device)
            batch_traj_lens = traj_lens[lb:ub]
            embeddings.append(self.model.forward(batch_traj_1d, batch_traj, batch_traj_lens).detach().cpu())
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings.numpy()

    # TODO: 实现faiss查询的函数
    def query_faiss(self, query_traj, k):
        """
        利用faiss框架加速查询过程
        :param query_traj:
        :param k:
        :return:
        """
        pass
