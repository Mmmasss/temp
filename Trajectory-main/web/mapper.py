from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, load_only
from typing import List

from model import Trajectory


class Mapper:
    def __init__(self, db_path="root:ma200399@localhost:3306/trajectory"):
        self.engine = create_engine(f"mysql+mysqldb://{db_path}", echo=True, future=True)

    def insert_trajectories(self, trajectories: List[Trajectory]):
        session = Session(self.engine)
        session.add_all(trajectories)
        session.commit()
        return 0

    def get_all_trajectories_all(self) -> List[Trajectory]:
        session = Session(self.engine)
        return session.query(Trajectory).all()

    def get_all_trajectories_points(self) -> List[Trajectory]:
        session = Session(self.engine)
        return session.query(Trajectory.id, Trajectory.points).all()

    def get_trajectories_points_by_time_range(self, start_time, end_time) -> List[Trajectory]:
        session = Session(self.engine)
        return session.query(Trajectory.id, Trajectory.points).filter(
            Trajectory.start_time < end_time).filter(start_time < Trajectory.end_time).all()

    def get_trajectories_embedding_by_time_range(self, start_time, end_time):
        session = Session(self.engine)
        return session.query(Trajectory.id, Trajectory.embedding).filter(
            Trajectory.start_time < end_time).filter(start_time < Trajectory.end_time).all()

    def get_all_trajectories_embedding(self) -> List[Trajectory]:
        session = Session(self.engine)
        return session.query(Trajectory.id, Trajectory.embedding).all()

    def get_trajectory_by_id(self, tid: int) -> Trajectory:
        session = Session(self.engine)
        return session.query(Trajectory).where(Trajectory.id == tid).first()

    def get_trajectory_by_id_list(self, id_list: List[int]) -> List[Trajectory]:
        session = Session(self.engine)
        trajectories = session.query(Trajectory).filter(Trajectory.id.in_(id_list)).all()
        trajectories.sort(key=lambda x: id_list.index(x.id))
        return trajectories

    def update_trajectory_embedding_by_id(self, tid: int, embedding: str):
        session = Session(self.engine)
        session.query(Trajectory).filter(Trajectory.id == tid).update({Trajectory.embedding: embedding})
        session.commit()
        return 0

    def update_trajectory_embedding_by_id_list(self, id_list: List[int], embeddings: List[str]):
        session = Session(self.engine)
        trajectories = session.query(Trajectory).filter(Trajectory.id.in_(id_list)).all()
        trajectories.sort(key=lambda x: id_list.index(x.id))
        for i, trajectory in enumerate(trajectories):
            trajectory.embedding = embeddings[i]
        session.commit()
        return 0
