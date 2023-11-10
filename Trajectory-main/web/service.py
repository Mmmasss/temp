from joblib import Parallel, delayed
import numpy as np
import traj_dist.distance as tdist
from typing import List, Tuple
import time
from efficient_solver import EfficientSolver
from mapper import Mapper
from model import Trajectory


class Service:
    def __init__(self):
        self.mapper = Mapper()
        mean_x, mean_y, std_x, std_y = self._get_mean_std()
        self.solver = EfficientSolver('E:\pythonProject\Trajectory-main\model\model_baseline_rank_12.034.pth', "cuda", mean_x, mean_y,
                                      std_x, std_y)
        self.job_nums = 6  # 传统查询调用的CPU核心数

    def _get_mean_std(self):
        """
        从数据库中获取轨迹的均值和标准差
        """
        mean_x, mean_y, std_x, std_y = 0, 0, 0, 0
        traj_list = self.mapper.get_all_trajectories_points()
        points_list = []
        for _, points in traj_list:
            points_list.extend(eval(points))
        mean_x = np.mean([point[0] for point in points_list])
        mean_y = np.mean([point[1] for point in points_list])
        std_x = np.std([point[0] for point in points_list])
        std_y = np.std([point[1] for point in points_list])
        return mean_x, mean_y, std_x, std_y

    def knn_query(self, query_traj: List[Tuple[float, float]], query_type: str, k: int,
                  time_range=None) -> Tuple[List[Trajectory], List[float], float]:
        """
        相似性轨迹检索的总入口
        :param query_traj: List[Tuple[float, float]]:query轨迹,经纬度坐标序列
        :param query_type: str:查询类型,字符串
        :param k: int k近邻
        :param time_range: List[int,int]: 时间戳切片,如[0,10]表示查询与时间戳0-10有交集的轨迹
        :return: tok_k_traj: 最相似的k个轨迹List[Trajectory]
                 top_k_sim: 最相似的k个轨迹的相似度
                 compute_time: 不含SQL操作的纯计算时间
        """
        top_k_sim = []
        top_k_id = []
        # 高效算法
        if query_type == "efficient_bf" or query_type == "efficient_faiss":
            # 根据是否考虑时间切片获取相应的轨迹列表
            if time_range:
                start_time, end_time = int(time_range[0]), int(time_range[1])
                traj_list = self.mapper.get_trajectories_embedding_by_time_range(start_time, end_time)
            else:
                traj_list = self.mapper.get_all_trajectories_embedding()
            st = time.time()
            compute_count = len(traj_list)
            id_list = [traj[0] for traj in traj_list]
            emb_arr = np.array([(eval(traj[1])) for traj in traj_list])
            # 获取query轨迹的embedding
            query_emb = self.solver.embed_trajectory(query_traj)
            if query_type == "efficient_bf":
                # 暴力计算query轨迹与所有轨迹的相似度
                query_emb = np.expand_dims(query_emb, axis=0)
                sim_matrix = self.solver.compute_similarity(query_emb, emb_arr).squeeze()
                sorted_idx = np.argsort(-sim_matrix)
                top_k_id = [id_list[i] for i in sorted_idx[:k]]
                top_k_sim = sim_matrix[sorted_idx[:k]].tolist()
                compute_time = time.time() - st
            else:
                # TODO: 用faiss查询
                pass
                compute_time = time.time() - st

        # 传统算法
        else:
            # 根据是否考虑时间切片获取相应的轨迹列表
            if time_range:
                start_time, end_time = int(time_range[0]), int(time_range[1])
                traj_list = self.mapper.get_trajectories_points_by_time_range(start_time, end_time)
            else:
                traj_list = self.mapper.get_all_trajectories_points()
            st = time.time()
            compute_count = len(traj_list)
            id_list = [traj[0] for traj in traj_list]
            points_list = [np.array(eval(traj[1])) for traj in traj_list]
            qt = np.array(query_traj)
            metric_func = getattr(tdist, query_type)

            # 定义相似度函数
            def cal_sim(tid, qt, traj: np.array):
                if query_type == "lcss":
                    dis = metric_func(qt, traj, eps=0.00029)
                else:
                    dis = metric_func(qt, traj)
                print(f"id:{tid}, dis:{dis}")
                sim = 1 - dis
                return tid, sim

            # 多核并行计算相似度
            res = Parallel(n_jobs=self.job_nums)(delayed(cal_sim)(tid, qt, t) for tid, t in zip(id_list, points_list))
            top_k_res = sorted(res, key=lambda x: x[1], reverse=True)[:k]
            top_k_id = [tid for tid, sim in top_k_res]
            top_k_sim = [sim for tid, sim in top_k_res]
            compute_time = time.time() - st
        tok_k_traj = self.mapper.get_trajectory_by_id_list(top_k_id)
        return tok_k_traj, top_k_sim, compute_time, compute_count

    def generate_embedding_all(self):
        """
        为整张表的轨迹(重新)生成embedding
        获取全表轨迹的id, points, 用solver生成embedding,再update回去
        """
        traj_list = self.mapper.get_all_trajectories_points()
        points_list = [eval(traj[1]) for traj in traj_list]
        id_list = [traj[0] for traj in traj_list]
        embeddings = self.solver.embed_trajectory_batch(points_list)
        embeddings = [str(embedding.tolist()) for embedding in embeddings]
        self.mapper.update_trajectory_embedding_by_id_list(id_list, embeddings)

    # TODO 向数据库中插入一批新轨迹并生成其embedding
    def insert_trajectories(self, trajectories):
        """
        向数据库中插入一批新轨迹并生成其embedding
        :param trajectories: List[List[float,float]]
        :return:
        """
        pass
