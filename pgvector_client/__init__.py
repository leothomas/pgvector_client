from __future__ import annotations

import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColumnDefinition:
    """Helper class for defining Postgres table columns"""

    name: str
    postgres_type: str
    dimension: Optional[int] = None

    def __init__(self, name: str, postgres_type: str, dimension: Optional[int] = None):
        self.name = name
        self.postgres_type = postgres_type
        if dimension:
            self.dimension = dimension

    def __str__(self):
        s = f'{self.name} {self.postgres_type}'
        if self.dimension:
            s = f'{s}({self.dimension})'
        return s


class IndexFamily(Enum):
    ivfflat = 'ivfflat'
    hnsw = 'hnsw'


class DistanceMetric(Enum):
    euclidean = 'euclidean'
    inner_product = 'inner_product'
    cosine = 'cosine'


DISTANCE_METRIC_TO_SEARCH_OP = {
    'euclidean': '<->',
    'inner_product': '<#>',
    'cosine': '<=>',
}

DISTANCE_METRIC_TO_INDEX_OP = {
    'euclidean': 'vector_l2_ops',
    'inner_product': 'vector_ip_ops',
    'cosine': 'vector_cosine_ops',
}


class VectorIndexBase:
    family: IndexFamily
    distance_metric: DistanceMetric
    name: str

    def __init__(
        self,
        distance_metric: DistanceMetric = DistanceMetric.euclidean,
    ):
        self.distance_metric = distance_metric


class VectorIndexIVFFlat(VectorIndexBase):
    family: IndexFamily = IndexFamily.ivfflat
    nlist: int

    def __init__(self, nlist: int, **kwargs):
        super().__init__(**kwargs)
        self.nlist = nlist
        self.name = f'vector_index_{self.family.value}_{self.distance_metric.value}_nl{self.nlist}'

    def build_params_string(self):
        return f'lists = {self.nlist}'

    def __repr__(self):
        return f'Index Definition <family: IVF, distance metric: {self.distance_metric.value}, build params(nlists={self.nlist})>'  # noqa


class VectorIndexHSNW(VectorIndexBase):
    family: IndexFamily = IndexFamily.hnsw
    m: int
    ef_construction: int

    def __init__(self, m: int = 16, ef_construction: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.ef_construction = ef_construction
        self.name = f'vector_index_{self.family.value}_{self.distance_metric.value}_m{m}_efc{ef_construction}'

    def build_params_string(self):
        return f'm = {self.m}, ef_construction = {self.ef_construction}'

    def __repr__(self):
        return f'Index Definition <family: HNSW, distance metric: {self.distance_metric.value}, build params(m={self.m}, ef_construction={self.ef_construction})'  # noqa


class VectorTable:
    def __init__(
        self,
        conninfo: Union[str, Dict],
        vector_column_dimension: int,
        schemaname: str = 'public',
        tablename: str = 'items',
        vector_column_name: str = 'vector',
        column_defs: List[ColumnDefinition] = [],
    ):
        if isinstance(conninfo, Dict):
            self.conninfo = ' '.join([f'{k}={v}' for k, v in conninfo.items()])
        else:
            self.conninfo = conninfo

        self.conn_pool = ConnectionPool(
            conninfo=self.conninfo,
            min_size=1,  # The minimum number of connection the pool will hold
            max_size=50,  # The maximum number of connections the pool will hold
            max_waiting=50000,  # Maximum number of requests that can be queued to the pool
            # Maximum time, in seconds, that a connection can stay unused in the pool before being
            # closed and the pool shrunk.
            max_idle=300,
            num_workers=3,  # Number of background worker threads used to maintain the pool state
            open=False,
            kwargs={'row_factory': dict_row},
        )

        self.conn_pool.open()

        self.schemaname = schemaname
        self.tablename = tablename

        # TODO: infer these values from the table, if exists
        self.vector_column_name = vector_column_name
        self.vector_column_dimension = vector_column_dimension

        self.column_defs = column_defs

    def __bootstrap_database(self):
        with self.conn_pool.connection() as conn:
            logger.info('Registering pgvector extension (if not exists)')
            conn.execute('CREATE EXTENSION IF NOT EXISTS vector')

        if self.schemaname != 'public':
            with self.conn_pool.connection() as conn:
                logger.info('Creating schema (if not exists)')
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS {self.schemaname};')

    def exists(self):
        with self.conn_pool.connection() as conn:
            result = conn.execute(
                f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = '{self.schemaname}' AND table_name = '{self.tablename}')",  # noqa
            )
            return result.fetchone()['exists']

    def truncate(self):
        logger.info(f'Dropping data from table {self.tablename}')
        with self.conn_pool.connection() as conn:
            conn.execute(f'TRUNCATE {self.schemaname}.{self.tablename}')

    def delete(self):
        logging.info(f'Dropping table {self.tablename} from database')
        with self.conn_pool.connection() as conn:
            conn.execute(f'DROP TABLE IF EXISTS {self.schemaname}.{self.tablename}')

    def create(
        self,
        recreate_if_exists: bool = False,
        disable_write_ahead_log: bool = False,
    ):
        self.__bootstrap_database()

        if self.exists() and recreate_if_exists:
            logger.info(
                f'Table exists, and recreation was requested. Dropping table {self.tablename}',
            )
            self.truncate()
            self.delete()

        with self.conn_pool.connection() as conn:
            logger.info(
                f"Creating table {self.tablename} (if not exists) with the following column definitions: {','.join(str(c) for c in self.column_defs)}",  # noqa
            )

            if any([c.name == 'id' for c in self.column_defs]):
                [id_column_def] = [c for c in self.column_defs if c.name == 'id']
                self.column_defs = [c for c in self.column_defs if c.name != 'id']
            else:
                id_column_def = ColumnDefinition(name='id', postgres_type='bigserial')

            if any(c.name == 'vector' for c in self.column_defs):
                logger.warn("Column name 'vector' is reserved for the vector column")
                self.column_defs = [c for c in self.column_defs if c.name != 'vector']

            column_defs = ','.join(str(c) for c in self.column_defs)
            column_defs = f', {column_defs}' if column_defs else ''

            conn.execute(
                f"CREATE {'UNLOGGED' if disable_write_ahead_log else ''} TABLE IF NOT EXISTS {self.schemaname}.{self.tablename} ({id_column_def} PRIMARY KEY, {self.vector_column_name} vector({self.vector_column_dimension}){column_defs})",  # noqa
            )

            id_column_def = ColumnDefinition(name='id', postgres_type='smallserial')

    def insert(
        self,
        records: List[Dict[str, Any]],
        abandon_on_insert_error: bool = False,
    ):
        if not self.exists():
            logger.error(
                f'Table {self.tablename} does not exist. Please create the table before loading data',
            )

        logger.info(f'Loading {len(records)} records into database')

        with self.conn_pool.connection() as conn:
            with conn.cursor() as cursor:
                with cursor.copy(
                    f"COPY {self.tablename} ({','.join(records[0].keys())}) FROM STDIN",
                ) as copy:
                    for record in records:
                        record[self.vector_column_name] = str(
                            record[self.vector_column_name],
                        )
                        try:
                            copy.write_row(tuple(record.values()))
                        except Exception:
                            if abandon_on_insert_error:
                                raise
                            logging.warn('Failed to insert record: ', record)

    @property
    def num_records(self):
        with self.conn_pool.connection() as conn:
            result = conn.execute(
                f'SELECT count(*) FROM {self.schemaname}.{self.tablename}',
            )
            return result.fetchone()['count']

    def index_exists(self, index: Union[VectorIndexIVFFlat, VectorIndexHSNW]):

        with self.conn_pool.connection() as conn:
            result = conn.execute(
                f"SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname = '{self.schemaname}' AND tablename = '{self.tablename}' AND indexname = '{index.name}')",  # noqa
            )
            return result.fetchone()['exists']

    def list_indexes(self):
        with self.conn_pool.connection() as conn:
            result = conn.execute(
                f"SELECT indexname FROM pg_indexes WHERE schemaname = '{self.schemaname}' AND tablename = '{self.tablename}'",  # noqa
            )
            return [r['index_name'] for r in result.fetchall()]

    def delete_index(self, index: Union[VectorIndexIVFFlat, VectorIndexHSNW]):
        with self.conn_pool.connection() as conn:
            conn.execute(f'DROP INDEX IF EXISTS {self.schemaname}.{index.name}')

    def create_index(
        self,
        index: Union[VectorIndexIVFFlat, VectorIndexHSNW],
        maintenance_work_mem: str = '1GB',
        max_parallel_workers: int = 1,
        recreate_if_exists: bool = False,
    ):

        if self.index_exists(index) and recreate_if_exists:
            logger.info(
                f'Index {index.name} exists, and recreation was requested. Dropping index',
            )
            self.delete_index(index)

        with self.conn_pool.connection() as conn:
            logger.info(
                f'Creating index {index.name} on table {self.tablename} using {index.family.value} index family',
            )
            conn.execute(f"SET LOCAL maintenance_work_mem = '{maintenance_work_mem}'")
            conn.execute(f'SET LOCAL max_parallel_workers = {max_parallel_workers}')
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS {index.name} ON {self.schemaname}.{self.tablename} USING {index.family.value} ({self.vector_column_name} {DISTANCE_METRIC_TO_INDEX_OP[index.distance_metric.value]}) WITH ({index.build_params_string()})',  # noqa
            )

    def get_index_build_status(self, index: Union[VectorIndexIVFFlat, VectorIndexHSNW]):
        with self.conn_pool.connection() as conn:
            result = conn.execute(
                f'SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 4) AS "% blocks done", round(100.0 * tuples_done / nullif(tuples_total , 0), 4) AS "% tuples done" FROM pg_stat_progress_create_index',  # noqa
            )
            return result.fetchone()

    # TODO: add PCA dimensionality reduction to ingestion and search
    # TODO: check if query has a covering index
    def search(
        self,
        query_vector: List[float],
        search_params: Dict = {},
        distance_metric: DistanceMetric = DistanceMetric.euclidean,
        num_results: Optional[int] = None,
        distance: Optional[float] = None,
        show_explain: bool = False,
    ):
        if not num_results and not distance:
            raise ValueError('Either limit or distance must be provided')

        # TODO: check if the query vector has the correct dimensions
        # TODO: check if the table has an index for the requested distance metric
        # TODO: create default search params per index

        with self.conn_pool.connection() as conn:

            for k, v in search_params.items():
                conn.execute(f'SET LOCAL {k} = {v}')

            distance_op = DISTANCE_METRIC_TO_SEARCH_OP[distance_metric.value]
            query_vector_str = str(query_vector)
            query = f"SELECT *, {self.vector_column_name} {distance_op} '{query_vector_str}' as _distance FROM {self.schemaname}.{self.tablename}"  # noqa

            if distance:
                query = f"{query} WHERE {self.vector_column_name} {distance_op} '{query_vector_str}' < {distance}"

            query = f"{query} ORDER BY {self.vector_column_name} {distance_op} '{query_vector_str}'"

            if num_results:
                query = f'{query} LIMIT {num_results}'

            if show_explain:
                logger.info(
                    '-----------------------EXPLAIN------------------------------',
                )
                for record in conn.execute(
                    f'EXPLAIN (ANALYZE,BUFFERS) {query}',
                ).fetchall():
                    logger.info(record)
                logger.info(
                    '-------------------END EXPLAIN------------------------------',
                )

            return conn.execute(query).fetchall()
