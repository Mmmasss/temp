import utils
from service import Service
from model import *
import unittest


class UnitTest(unittest.TestCase):

    def setUp(self):
        self.q_traj = [(104.06075, 30.65499), (104.06075, 30.65515), (104.06075, 30.65533), (104.06074, 30.6556),
                       (104.06078, 30.65593),
                       (104.06081, 30.65621), (104.0608, 30.65644), (104.06079, 30.65661), (104.06079, 30.65674),
                       (104.06079, 30.65691),
                       (104.06079, 30.65713), (104.06079, 30.65736), (104.06079, 30.65743), (104.06078, 30.65764),
                       (104.06078, 30.65783),
                       (104.06078, 30.65804), (104.06066, 30.6583), (104.06045, 30.65829), (104.06024, 30.65829),
                       (104.05998, 30.65829),
                       (104.05919, 30.65828), (104.05891, 30.65829), (104.05871, 30.65832), (104.0584, 30.65843),
                       (104.05814, 30.65861),
                       (104.05797, 30.65875), (104.05786, 30.65886), (104.05778, 30.65893), (104.05765, 30.65905),
                       (104.05751, 30.65918),
                       (104.05742, 30.65926), (104.05731, 30.65936), (104.05721, 30.65945), (104.05707, 30.65957),
                       (104.05693, 30.6597),
                       (104.05674, 30.65987), (104.05655, 30.66004), (104.05643, 30.66015), (104.05622, 30.66033),
                       (104.05599, 30.66049),
                       (104.05574, 30.66064), (104.05547, 30.66074), (104.05511, 30.66082), (104.05484, 30.66086),
                       (104.05451, 30.66087),
                       (104.05423, 30.66088), (104.05399, 30.66087), (104.05389, 30.66086), (104.05357, 30.66085),
                       (104.05347, 30.66084),
                       (104.05307, 30.66084), (104.05278, 30.66084), (104.05248, 30.66085), (104.05219, 30.66089),
                       (104.05147, 30.66108),
                       (104.05122, 30.66116), (104.05097, 30.66124), (104.05066, 30.66135), (104.05035, 30.66145),
                       (104.05014, 30.66153),
                       (104.04982, 30.66165), (104.04951, 30.66177), (104.04921, 30.66189), (104.04893, 30.66202),
                       (104.04866, 30.66216),
                       (104.04839, 30.66228), (104.04833, 30.66231), (104.04826, 30.66234), (104.04797, 30.66248),
                       (104.0478, 30.66256),
                       (104.04758, 30.66266), (104.04743, 30.66273), (104.0473, 30.66279), (104.04696, 30.66296),
                       (104.04687, 30.66301),
                       (104.04676, 30.66306), (104.04668, 30.6631), (104.04651, 30.66319), (104.04632, 30.66329),
                       (104.04616, 30.66337),
                       (104.04607, 30.66342), (104.04573, 30.66359), (104.04569, 30.66361), (104.04562, 30.66364)]
        self.test_trajs = [[(104.04668, 30.65522), (104.0465, 30.65552), (104.04637, 30.65573), (104.04619, 30.65602),
                            (104.04605, 30.65624), (104.04595, 30.65642), (104.0458, 30.65667), (104.04562, 30.65698),
                            (104.0455, 30.65715), (104.04536, 30.65735), (104.04518, 30.65756), (104.04488, 30.65781),
                            (104.04465, 30.65795), (104.04442, 30.65805), (104.04424, 30.6581)],
                           [(104.04668, 30.65522), (104.0465, 30.65552), (104.04637, 30.65573), (104.04619, 30.65602),
                            (104.04605, 30.65624), (104.04595, 30.65642), (104.0458, 30.65667), (104.04562, 30.65698),
                            (104.0455, 30.65715), (104.04536, 30.65735)]]
        self.service = Service()
        print('setUp...')

    def test_knn_query_efficient_bf(self):
        res, _ = self.service.knn_query(query_traj=self.q_traj, k=3, query_type="efficient_bf",
                                        time_range=[1478063519, 1478064044])
        print(res)

    def test_knn_query_tradition(self):
        res2, _ = self.service.knn_query(query_traj=self.q_traj, k=3, query_type="discret_frechet",
                                         time_range=[1478063519, 1478064044])
        print(res2)

    def test_embed_trajectory_batch(self):
        emb = self.service.solver.embed_trajectory_batch(self.test_trajs)
        print(emb.shape)

    def test_generate_embedding_all(self):
        self.service.generate_embedding_all()

    def test_get_all_trajectories_embedding(self):
        res = self.service.mapper.get_all_trajectories_embedding()
        print(res)

    def test_get_all_trajectories_points(self):
        res = self.service.mapper.get_all_trajectories_points()
        print(res)

    def test__get_mean_std(self):
        res = self.service.mapper._get_mean_std()
        print(res)

    def test_get_trajectory_by_id_list(self):
        res = self.service.mapper.get_trajectory_by_id_list([4, 2, 1, 3])

    def test_update_trajectory_embedding_by_id_list(self):
        self.service.mapper.update_trajectory_embedding_by_id_list([4, 2, 1, 3], ["44444", "2222", "11111", "33333"])