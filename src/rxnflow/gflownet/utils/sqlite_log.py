import sqlite3
from collections.abc import Iterable
from pathlib import Path

from ..types import GFNEnvironmentContext, Traj


class SQLiteLogHook:
    def __init__(self, log_path: str, ctx: GFNEnvironmentContext) -> None:
        self.log = None  # Initialized in __call__
        self.data_labels = None  # Initialized in __call__
        self.log_path: Path = Path(log_path)
        self.ctx: GFNEnvironmentContext = ctx

    def __call__(self, trajs: list[Traj]):
        if self.log is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = SQLiteLog()
            self.log.connect(str(self.log_path), check_same_thread=False)

        objs: list[str] = [
            self.ctx.object_to_log_repr(t["result"]) if t["is_valid"] else ""
            for t in trajs
        ]
        workflows: list[str] = [
            self.ctx.traj_to_workflow(t["traj"]) if t["is_valid"] else "" for t in trajs
        ]
        traj_str: list[str] = [
            self.ctx.traj_to_log_repr(t["traj"]) if t["is_valid"] else "" for t in trajs
        ]
        rewards: list[float] = [t["reward"] for t in trajs]

        data = [
            [objs[i], rewards[i], workflows[i], traj_str[i]] for i in range(len(trajs))
        ]
        if self.data_labels is None:
            self.data_labels = ["smi", "r", "workflow", "traj"]

        self.log.insert_many(data, self.data_labels)
        return {}


class SQLiteLog:
    def __init__(self, timeout=300):
        """Creates a log instance, but does not connect it to any db."""
        self.is_connected = False
        self.db = None
        self.timeout = timeout

    def connect(self, db_path: str, **kwargs):
        """Connects to db_path

        Parameters
        ----------
        db_path: str
            The sqlite3 database path. If it does not exist, it will be created.
        """
        self.db = sqlite3.connect(db_path, timeout=self.timeout, **kwargs)
        cur = self.db.cursor()
        self._has_results_table = len(
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='results'"
            ).fetchall()
        )
        cur.close()

    def _make_results_table(self, types, names):
        assert self.db is not None
        type_map = {str: "text", float: "real", int: "real"}
        col_str = ", ".join(
            f"{name} {type_map[t]}" for t, name in zip(types, names, strict=False)
        )
        cur = self.db.cursor()
        cur.execute(f"create table results ({col_str})")
        self._has_results_table = True
        cur.close()

    def insert_many(self, rows, column_names):
        assert self.db is not None
        assert all(
            [isinstance(x, str) or not isinstance(x, Iterable) for x in rows[0]]
        ), "rows must only contain scalars"
        if not self._has_results_table:
            self._make_results_table([type(i) for i in rows[0]], column_names)
        cur = self.db.cursor()
        cur.executemany(
            f"insert into results values ({','.join('?' * len(rows[0]))})", rows
        )  # nosec
        cur.close()
        self.db.commit()

    def __del__(self):
        if self.db is not None:
            self.db.close()


def read_all_results(path):
    # E402: module level import not at top of file, but pandas is an optional dependency
    import pandas as pd  # noqa: E402

    df = pd.read_sql_query(
        "SELECT * FROM results", sqlite3.connect(f"file:{path}/generated_objs.db?mode=ro")
    )
    return df.sort_index().reset_index(drop=True)
