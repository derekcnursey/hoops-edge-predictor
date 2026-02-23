"""Read Parquet data from the hoops-edge S3 lakehouse."""

from __future__ import annotations

import io
from typing import Optional

import boto3
import pyarrow as pa
import pyarrow.parquet as pq

from . import config


def _s3_client():
    return boto3.client("s3", region_name=config.S3_REGION)


def list_parquet_keys(prefix: str, bucket: str = config.S3_BUCKET) -> list[str]:
    """List all .parquet file keys under a given S3 prefix."""
    client = _s3_client()
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    return keys


def read_parquet_table(
    keys: list[str],
    bucket: str = config.S3_BUCKET,
    columns: Optional[list[str]] = None,
) -> pa.Table:
    """Read and concatenate Parquet files from S3 into a single PyArrow table."""
    client = _s3_client()
    tables: list[pa.Table] = []
    for key in keys:
        resp = client.get_object(Bucket=bucket, Key=key)
        data = resp["Body"].read()
        tbl = pq.read_table(io.BytesIO(data), columns=columns)
        tables.append(tbl)
    if not tables:
        return pa.table({})
    try:
        return pa.concat_tables(tables, promote_options="permissive")
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        # Fallback: cast conflicting types to float64 or string
        return _concat_with_type_promotion(tables)


def _concat_with_type_promotion(tables: list[pa.Table]) -> pa.Table:
    """Concatenate tables with schema mismatches by promoting types."""
    col_types: dict[str, pa.DataType] = {}
    for tbl in tables:
        for field in tbl.schema:
            existing = col_types.get(field.name)
            if existing is None:
                col_types[field.name] = field.type
            elif existing != field.type:
                numeric = {pa.int8(), pa.int16(), pa.int32(), pa.int64(),
                           pa.float16(), pa.float32(), pa.float64()}
                if existing in numeric and field.type in numeric:
                    col_types[field.name] = pa.float64()
                else:
                    col_types[field.name] = pa.string()
    unified = pa.schema([(n, t) for n, t in col_types.items()])
    aligned: list[pa.Table] = []
    for tbl in tables:
        cols = {}
        for field in unified:
            if field.name in tbl.column_names:
                col = tbl.column(field.name)
                if col.type != field.type:
                    col = col.cast(field.type, safe=False)
                cols[field.name] = col
            else:
                cols[field.name] = pa.nulls(tbl.num_rows, type=field.type)
        aligned.append(pa.table(cols))
    return pa.concat_tables(aligned)


def _get_latest_asof_prefix(base_prefix: str, bucket: str = config.S3_BUCKET) -> Optional[str]:
    """Find the latest asof= partition under a base prefix."""
    client = _s3_client()
    resp = client.list_objects_v2(
        Bucket=bucket, Prefix=base_prefix, Delimiter="/"
    )
    prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    # Filter to asof= partitions and sort to get latest
    asof_prefixes = sorted(
        [p for p in prefixes if "asof=" in p],
        reverse=True,
    )
    return asof_prefixes[0] if asof_prefixes else None


def read_silver_table(
    table_name: str,
    season: Optional[int] = None,
    latest_only: bool = False,
) -> pa.Table:
    """Read a silver-layer table from S3.

    Args:
        table_name: e.g. 'fct_games', 'fct_pbp_game_teams_flat'
        season: Filter to season=YYYY partition if provided.
        latest_only: If True, only read the latest asof= partition.

    Returns:
        PyArrow Table with all rows from matching partitions.
    """
    base = f"{config.SILVER_PREFIX}/{table_name}/"
    if season is not None:
        base = f"{config.SILVER_PREFIX}/{table_name}/season={season}/"

    if latest_only:
        latest = _get_latest_asof_prefix(base)
        if latest is None:
            return pa.table({})
        keys = list_parquet_keys(latest)
    else:
        keys = list_parquet_keys(base)

    if not keys:
        return pa.table({})
    return read_parquet_table(keys)


def read_gold_table(table_name: str, season: Optional[int] = None) -> pa.Table:
    """Read a gold-layer table from S3."""
    base = f"{config.GOLD_PREFIX}/{table_name}/"
    if season is not None:
        base = f"{config.GOLD_PREFIX}/{table_name}/season={season}/"
    keys = list_parquet_keys(base)
    if not keys:
        return pa.table({})
    return read_parquet_table(keys)


def get_column(table: pa.Table, *candidates: str) -> list:
    """Get a column by trying multiple name candidates. Returns Python list."""
    for col in candidates:
        if col in table.column_names:
            return table.column(col).to_pylist()
    return [None] * table.num_rows


def write_parquet_to_s3(
    table: pa.Table,
    key: str,
    bucket: str = config.S3_BUCKET,
) -> None:
    """Write a PyArrow table to S3 as a single Parquet file."""
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    client = _s3_client()
    client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
