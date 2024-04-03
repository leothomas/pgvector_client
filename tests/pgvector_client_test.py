from __future__ import annotations

import psycopg
import pytest
from psycopg.rows import dict_row

from pgvector_client.client import VectorIndexIVFFlat
from pgvector_client.client import VectorTable

CONNINFO = 'host=localhost port=5432 dbname=postgres user=postgres password=password'


# TODO: Test non-default table options (tablename, schemaname, vector_column_name, etc.)
# TODO: Test non-default index options for both IVFFlat and HNSW (incl. distance metrics)
def test_create_table():
    table = VectorTable(
        conninfo=CONNINFO,
        vector_column_dimension=128,
    )
    table.create()

    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM information_schema.tables WHERE table_name = 'items'",
            )
            assert cur.fetchone() is not None


def test_delete_table():
    table = VectorTable(
        conninfo=CONNINFO,
        vector_column_dimension=128,
    )
    table.create()

    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM information_schema.tables WHERE table_name = 'items'",
            )
            assert cur.fetchone() is not None

    table.delete()

    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM information_schema.tables WHERE table_name = 'items'",
            )
            assert cur.fetchone() is None


def test_table_exists():
    table = VectorTable(
        conninfo=CONNINFO,
        vector_column_dimension=128,
    )
    table.delete()
    assert table.exists() is False
    table.create()
    assert table.exists() is True


def test_insert_data():
    table = VectorTable(
        conninfo=CONNINFO,
        vector_column_dimension=128,
    )
    table.delete()
    table.create()
    table.insert(records=[{'vector': [0 for _ in range(128)]} for _ in range(100)])
    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.items')
            assert cur.fetchone()['count'] > 0


@pytest.fixture
def table_with_data():
    table = VectorTable(
        conninfo=CONNINFO,
        vector_column_dimension=128,
    )
    table.delete()
    table.create()
    table.insert(records=[{'vector': [0 for _ in range(128)]} for _ in range(100)])
    return table


def test_table_creation_skips_if_already_exists(table_with_data):
    table = table_with_data
    table.create()
    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.items')
            assert cur.fetchone()['count'] == 100


def test_table_creation_with_recreate_if_exists(table_with_data):
    table = table_with_data
    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.items')
            assert cur.fetchone()['count'] == 100
    table.create(recreate_if_exists=True)
    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.items')
            assert cur.fetchone()['count'] == 0


def test_truncate_table(table_with_data):
    table = table_with_data
    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.items')
            assert cur.fetchone()['count'] == 100
    table.truncate()
    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.items')
            assert cur.fetchone()['count'] == 0


def test_create_index(table_with_data):
    table = table_with_data
    index = VectorIndexIVFFlat(nlist=100)
    table.create_index(index)
    with psycopg.connect(CONNINFO, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM pg_indexes WHERE tablename = 'items' AND indexname = '{index.name}'",
            )
            assert cur.fetchone() is not None


def test_index_exists(table_with_data):
    table = table_with_data
    index = VectorIndexIVFFlat(nlist=100)
    assert table.index_exists(index) is False
    table.create_index(index)
    assert table.index_exists(index) is True


def test_delete_index(table_with_data):
    table = table_with_data
    index = VectorIndexIVFFlat(nlist=100)
    table.create_index(index)
    assert table.index_exists(index) is True
    table.delete_index(index)
    assert table.index_exists(index) is False


def test_search_by_num_results(table_with_data):
    table = table_with_data
    records = table.search(query_vector=[0 for _ in range(128)], num_results=10)
    assert len(records) == 10


def test_search_by_distance(table_with_data):
    table = table_with_data
    records = table.search(query_vector=[0 for _ in range(128)], distance=0.5)
    assert len(records) == 100


def test_search_fails_without_both_num_results_and_distance(table_with_data):
    table = table_with_data
    with pytest.raises(ValueError):
        table.search(query_vector=[0 for _ in range(128)])
