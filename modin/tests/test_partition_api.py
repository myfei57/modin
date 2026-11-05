# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.distributed.dataframe.pandas import from_partitions, unwrap_partitions
from modin.pandas.accessor import PartitionAxisStats, PartitionLayout
from modin.pandas.indexing import compute_sliced_len
from modin.tests.pandas.utils import df_equals, test_data

PartitionClass = (
    FactoryDispatcher.get_factory().io_cls.frame_cls._partition_mgr_cls._partition_class
)

if Engine.get() == "Ray":
    from modin.core.execution.ray.common import RayWrapper
    from modin.core.execution.ray.common.utils import ObjectIDType

    put_func = RayWrapper.put
    get_func = RayWrapper.materialize
    is_future = lambda obj: isinstance(obj, ObjectIDType)  # noqa: E731
elif Engine.get() == "Dask":
    from distributed import Future

    from modin.core.execution.dask.common import DaskWrapper

    # Looks like there is a key collision;
    # https://github.com/dask/distributed/issues/3703#issuecomment-619446739
    # recommends to use `hash=False`. Perhaps this should be the default value of `put`.
    put_func = lambda obj: DaskWrapper.put(obj, hash=False)  # noqa: E731
    get_func = DaskWrapper.materialize
    is_future = lambda obj: isinstance(obj, Future)  # noqa: E731
elif Engine.get() == "Unidist":
    from unidist import is_object_ref

    from modin.core.execution.unidist.common import UnidistWrapper

    put_func = UnidistWrapper.put
    get_func = UnidistWrapper.materialize
    is_future = is_object_ref
elif Engine.get() == "Python":
    put_func = lambda x: x  # noqa: E731
    get_func = lambda x: x  # noqa: E731
    is_future = lambda obj: isinstance(obj, object)  # noqa: E731
else:
    raise NotImplementedError(
        f"'{Engine.get()}' engine is not supported by these test suites"
    )

NPartitions.put(4)
# HACK: implicit engine initialization (Modin issue #2989)
pd.DataFrame([])


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("reverse_index", [True, False])
@pytest.mark.parametrize("reverse_columns", [True, False])
def test_unwrap_partitions(axis, reverse_index, reverse_columns):
    data = test_data["int_data"]

    def get_df(lib, data):
        df = lib.DataFrame(data)
        if reverse_index:
            df.index = df.index[::-1]
        if reverse_columns:
            df.columns = df.columns[::-1]
        return df

    df = get_df(pd, data)
    # `df` should not have propagated the index and column updates to its
    # partitions yet. The partitions of `expected_df` should have the updated
    # metadata because we construct `expected_df` directly from the updated
    # pandas dataframe.
    expected_df = pd.DataFrame(get_df(pandas, data))
    expected_partitions = expected_df._query_compiler._modin_frame._partitions
    if axis is None:
        actual_partitions = np.array(unwrap_partitions(df, axis=axis))
        assert expected_partitions.shape == actual_partitions.shape
        for row_idx in range(expected_partitions.shape[0]):
            for col_idx in range(expected_partitions.shape[1]):
                df_equals(
                    get_func(expected_partitions[row_idx][col_idx].list_of_blocks[0]),
                    get_func(actual_partitions[row_idx][col_idx]),
                )
    else:
        expected_axis_partitions = (
            expected_df._query_compiler._modin_frame._partition_mgr_cls.axis_partition(
                expected_partitions, axis ^ 1
            )
        )
        expected_axis_partitions = [
            axis_partition.force_materialization().unwrap(squeeze=True)
            for axis_partition in expected_axis_partitions
        ]
        actual_axis_partitions = unwrap_partitions(df, axis=axis)
        assert len(expected_axis_partitions) == len(actual_axis_partitions)
        for item_idx in range(len(expected_axis_partitions)):
            if Engine.get() in ["Ray", "Dask", "Unidist"]:
                df_equals(
                    get_func(expected_axis_partitions[item_idx]),
                    get_func(actual_axis_partitions[item_idx]),
                )


def test_unwrap_virtual_partitions():
    # see #5164 for details
    data = test_data["int_data"]
    df = pd.DataFrame(data)
    virtual_partitioned_df = pd.concat([df] * 10)
    actual_partitions = np.array(unwrap_partitions(virtual_partitioned_df, axis=None))
    expected_df = pd.concat([pd.DataFrame(data)] * 10)
    expected_partitions = expected_df._query_compiler._modin_frame._partitions
    assert expected_partitions.shape == actual_partitions.shape

    for row_idx in range(expected_partitions.shape[0]):
        for col_idx in range(expected_partitions.shape[1]):
            df_equals(
                get_func(
                    expected_partitions[row_idx][col_idx]
                    .force_materialization()
                    .list_of_blocks[0]
                ),
                get_func(actual_partitions[row_idx][col_idx]),
            )


@pytest.mark.parametrize("column_widths", [None, "column_widths"])
@pytest.mark.parametrize("row_lengths", [None, "row_lengths"])
@pytest.mark.parametrize("columns", [None, "columns"])
@pytest.mark.parametrize("index", [None, "index"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_from_partitions(axis, index, columns, row_lengths, column_widths):
    data = test_data["int_data"]
    df1, df2 = pandas.DataFrame(data), pandas.DataFrame(data)
    num_rows, num_cols = df1.shape
    expected_df = pandas.concat([df1, df2], axis=1 if axis is None else axis)

    index = expected_df.index if index == "index" else None
    columns = expected_df.columns if columns == "columns" else None
    row_lengths = (
        None
        if row_lengths is None
        else [num_rows, num_rows] if axis == 0 else [num_rows]
    )
    column_widths = (
        None
        if column_widths is None
        else [num_cols] if axis == 0 else [num_cols, num_cols]
    )
    futures = []
    if axis is None:
        futures = [[put_func(df1), put_func(df2)]]
    else:
        futures = [put_func(df1), put_func(df2)]
    actual_df = from_partitions(
        futures,
        axis,
        index=index,
        columns=columns,
        row_lengths=row_lengths,
        column_widths=column_widths,
    )
    df_equals(expected_df, actual_df)


@pytest.mark.parametrize("columns", ["original_col", "new_col"])
@pytest.mark.parametrize("index", ["original_idx", "new_idx"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_from_partitions_mismatched_labels(axis, index, columns):
    expected_df = pd.DataFrame(test_data["int_data"])
    partitions = unwrap_partitions(expected_df, axis=axis)

    index = (
        expected_df.index
        if index == "original_idx"
        else [f"row{i}" for i in expected_df.index]
    )
    columns = (
        expected_df.columns
        if columns == "original_col"
        else [f"col{i}" for i in expected_df.columns]
    )

    expected_df.index = index
    expected_df.columns = columns
    actual_df = from_partitions(partitions, axis=axis, index=index, columns=columns)
    df_equals(expected_df, actual_df)


@pytest.mark.parametrize("row_labels", [[0, 2], slice(None)])
@pytest.mark.parametrize("col_labels", [[0, 2], slice(None)])
@pytest.mark.parametrize("is_length_future", [False, True])
@pytest.mark.parametrize("is_width_future", [False, True])
def test_mask_preserve_cache(row_labels, col_labels, is_length_future, is_width_future):
    def deserialize(obj):
        if is_future(obj):
            return get_func(obj)
        return obj

    def compute_length(indices, length):
        if not isinstance(indices, slice):
            return len(indices)
        return compute_sliced_len(indices, length)

    df = pandas.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 10, 11, 12]})
    obj_id = put_func(df)

    partition_shape = [
        put_func(len(df)) if is_length_future else len(df),
        put_func(len(df.columns)) if is_width_future else len(df.columns),
    ]

    source_partition = PartitionClass(obj_id, *partition_shape)
    masked_partition = source_partition.mask(
        row_labels=row_labels, col_labels=col_labels
    )

    expected_length = compute_length(row_labels, len(df))
    expected_width = compute_length(col_labels, len(df.columns))

    # Check that the cache is preserved
    assert expected_length == deserialize(masked_partition._length_cache)
    assert expected_width == deserialize(masked_partition._width_cache)
    # Check that the cache is interpreted properly
    assert expected_length == masked_partition.length()
    assert expected_width == masked_partition.width()
    # Recompute shape explicitly to check that the cached data was correct
    expected_length, expected_width = [
        masked_partition._length_cache,
        masked_partition._width_cache,
    ]
    masked_partition._length_cache = None
    masked_partition._width_cache = None
    assert expected_length == masked_partition.length()
    assert expected_width == masked_partition.width()


def _build_uneven_partitioned_frame():
    """Build a 16x8 frame with uneven row/column partition sizes."""
    row_sizes = [1, 3, 10, 2]
    col_sizes = [2, 5, 1]
    pdf = pandas.DataFrame(
        np.arange(16 * 8).reshape(16, 8),
        index=[f"row{i}" for i in range(16)],
        columns=[f"col{i}" for i in range(8)],
    )
    row_bounds = np.cumsum([0] + row_sizes)
    col_bounds = np.cumsum([0] + col_sizes)
    futures = [
        [
            put_func(
                pdf.iloc[
                    row_bounds[i] : row_bounds[i + 1],
                    col_bounds[j] : col_bounds[j + 1],
                ]
            )
            for j in range(len(col_sizes))
        ]
        for i in range(len(row_sizes))
    ]
    df = from_partitions(
        futures,
        axis=None,
        index=pdf.index,
        columns=pdf.columns,
        row_lengths=row_sizes,
        column_widths=col_sizes,
    )
    return pdf, df, row_sizes, col_sizes


def test_partition_layout_uneven():
    pdf, df, row_sizes, col_sizes = _build_uneven_partitioned_frame()
    layout = df.modin.partition_layout()

    assert isinstance(layout, PartitionLayout)

    assert layout.rows == PartitionAxisStats(
        sizes=tuple(row_sizes), num_partitions=4, min=1, max=10, total=16
    )
    assert layout.columns == PartitionAxisStats(
        sizes=tuple(col_sizes), num_partitions=3, min=1, max=5, total=8
    )

    # sizes are reported in the physical order (not sorted/rebalanced)
    assert layout.rows.sizes == tuple(row_sizes)
    assert layout.columns.sizes == tuple(col_sizes)
    # the uneven partitioning is detectable
    assert layout.rows.min < layout.rows.max
    assert layout.columns.min < layout.columns.max

    # totals agree with the frame shape and the data itself is intact
    assert layout.rows.total == len(df) == len(pdf)
    assert layout.columns.total == df.shape[1] == pdf.shape[1]
    df_equals(df, pdf)

    # repeated diagnostics return the same snapshot
    assert df.modin.partition_layout() == layout


def test_partition_layout_cache_miss():
    pdf, df, row_sizes, col_sizes = _build_uneven_partitioned_frame()
    modin_frame = df._query_compiler._modin_frame

    # simulate a full size cache miss at both the frame and partition level
    modin_frame._row_lengths_cache = None
    modin_frame._column_widths_cache = None
    for block in modin_frame._partitions.flatten():
        block._length_cache = None
        block._width_cache = None

    layout = df.modin.partition_layout()

    # sizes were computed through the standard mechanism, not taken from a cache
    assert modin_frame._row_lengths_cache == row_sizes
    assert modin_frame._column_widths_cache == col_sizes
    assert layout.rows.sizes == tuple(row_sizes)
    assert layout.columns.sizes == tuple(col_sizes)
    assert layout.rows.total == len(pdf)
    assert layout.columns.total == len(pdf.columns)
    df_equals(df, pdf)


def test_partition_layout_is_read_only():
    pdf, df, _, _ = _build_uneven_partitioned_frame()
    modin_frame = df._query_compiler._modin_frame

    # drop all metadata caches so that diagnostics has to compute everything
    modin_frame._row_lengths_cache = None
    modin_frame._column_widths_cache = None
    modin_frame.set_index_cache(None)
    modin_frame.set_columns_cache(None)
    for block in modin_frame._partitions.flatten():
        block._length_cache = None
        block._width_cache = None

    partitions_before = modin_frame._partitions
    blocks_before = modin_frame._partitions.copy()

    df.modin.partition_layout()

    # no repartitioning: the very same partition grid and objects are kept
    assert modin_frame._partitions is partitions_before
    assert modin_frame._partitions.shape == partitions_before.shape
    for i in range(partitions_before.shape[0]):
        for j in range(partitions_before.shape[1]):
            assert modin_frame._partitions[i, j] is blocks_before[i, j]

    # data, index and columns are unchanged
    assert list(df.index) == list(pdf.index)
    assert list(df.columns) == list(pdf.columns)
    df_equals(df, pdf)


@pytest.mark.parametrize(
    "frame_factory",
    [
        lambda: pd.DataFrame(),
        lambda: pd.DataFrame(columns=["a", "b", "c"]),
        lambda: pd.DataFrame(index=pandas.RangeIndex(5)),
    ],
)
def test_partition_layout_empty_frames(frame_factory):
    df = frame_factory()
    modin_frame = df._query_compiler._modin_frame
    layout = df.modin.partition_layout()

    assert isinstance(layout, PartitionLayout)

    # statistics describe the frame's partition sizes, in their stored order
    assert layout.rows.sizes == tuple(modin_frame.row_lengths)
    assert layout.columns.sizes == tuple(modin_frame.column_widths)

    # totals always equal the frame shape even for empty/deferred frames
    assert layout.rows.total == len(df)
    assert layout.columns.total == df.shape[1]
    assert sum(layout.rows.sizes) == layout.rows.total
    assert sum(layout.columns.sizes) == layout.columns.total

    # explicit min/max statistics without division by zero
    for axis_stats in (layout.rows, layout.columns):
        assert axis_stats.num_partitions == len(axis_stats.sizes)
        if axis_stats.num_partitions == 0:
            assert axis_stats.sizes == ()
            assert axis_stats.min is None
            assert axis_stats.max is None
            assert axis_stats.total == 0
        else:
            assert axis_stats.min == min(axis_stats.sizes)
            assert axis_stats.max == max(axis_stats.sizes)

    # the call works again and the empty frame content is intact
    assert df.modin.partition_layout() == layout
    df_equals(df, df.modin.to_pandas())


def test_partition_layout_empty_partition_grid():
    # an internal frame whose physical partition grid is completely empty
    frame_cls = FactoryDispatcher.get_factory().io_cls.frame_cls
    empty_grid = np.empty((0, 0), dtype=object)
    query_compiler = PandasQueryCompiler(frame_cls(empty_grid))
    df = pd.DataFrame(query_compiler=query_compiler)

    layout = df.modin.partition_layout()

    empty_stats = PartitionAxisStats(
        sizes=(), num_partitions=0, min=None, max=None, total=0
    )
    assert layout.rows == empty_stats
    assert layout.columns == empty_stats
    # diagnostics must not populate the empty grid with partitions
    assert query_compiler._modin_frame._partitions.shape == (0, 0)


def test_partition_layout_series():
    series = pd.Series(list(range(10)))
    layout = series.modin.partition_layout()

    assert isinstance(layout, PartitionLayout)
    assert sum(layout.rows.sizes) == layout.rows.total == len(series) == 10
    assert layout.columns.total == 1
    assert layout.rows.num_partitions >= 1
    pandas.testing.assert_series_equal(
        series.modin.to_pandas(), pandas.Series(range(10))
    )


def test_partition_layout_totals_after_filtering():
    data = test_data["int_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    filtered = modin_df[modin_df.iloc[:, 0] > modin_df.iloc[:, 0].min()]
    expected = pandas_df[pandas_df.iloc[:, 0] > pandas_df.iloc[:, 0].min()]

    layout = filtered.modin.partition_layout()
    assert layout.rows.total == len(filtered) == len(expected)
    assert layout.columns.total == filtered.shape[1] == expected.shape[1]
    assert len(layout.rows.sizes) == layout.rows.num_partitions
    assert len(layout.columns.sizes) == layout.columns.num_partitions
    df_equals(filtered, expected)
