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

"""Tests for ``modin_df.modin.to_pandas_batches``."""

import numpy as np
import pandas
import pandas.testing as tm
import pytest

import modin.pandas as pd
from modin.config import NPartitions

NPartitions.put(4)


def _collect(modin_df, batch_size):
    return list(modin_df.modin.to_pandas_batches(batch_size=batch_size))


@pytest.mark.parametrize("batch_size", [1, 3, 7, 10, 32, 100])
def test_batch_sizes_and_order(batch_size):
    pandas_df = pandas.DataFrame(
        {
            "a": np.arange(32, dtype="int64"),
            "b": np.arange(32, dtype="float64") + 0.5,
            "c": [f"val{i}" for i in range(32)],
        },
        index=[2 * i for i in range(32)],
    )
    modin_df = pd.DataFrame(pandas_df)

    batches = _collect(modin_df, batch_size)

    assert all(isinstance(batch, pandas.DataFrame) for batch in batches)
    assert all(len(batch) <= batch_size for batch in batches)
    assert sum(len(batch) for batch in batches) == len(pandas_df)
    # every batch except the last must be full
    assert all(len(batch) == batch_size for batch in batches[:-1])
    remainder = len(pandas_df) % batch_size
    expected_last = remainder if remainder else min(batch_size, len(pandas_df))
    assert len(batches[-1]) == expected_last

    # row order and row index are preserved batch by batch
    collected = pandas.concat(batches)
    tm.assert_frame_equal(collected, modin_df.modin.to_pandas())
    tm.assert_frame_equal(collected, pandas_df)


def test_batch_size_larger_than_data():
    pandas_df = pandas.DataFrame({"a": [1, 2, 3]})
    batches = _collect(pd.DataFrame(pandas_df), 1000)
    assert len(batches) == 1
    tm.assert_frame_equal(batches[0], pandas_df)


def test_duplicate_columns_and_dtypes_preserved():
    pandas_df = pandas.DataFrame(
        {
            "x": np.arange(16, dtype="int32"),
            "x_dup": np.arange(16, 0, -1, dtype="float32"),
            "y": [f"s{i}" for i in range(16)],
        }
    )
    pandas_df.columns = ["x", "x", "y"]
    modin_df = pd.DataFrame(pandas_df)

    batches = _collect(modin_df, 5)
    for batch in batches:
        assert batch.columns.tolist() == ["x", "x", "y"]
        assert list(batch.dtypes) == list(pandas_df.dtypes)

    tm.assert_frame_equal(pandas.concat(batches), pandas_df)


def test_custom_and_duplicated_index_preserved():
    index = pandas.Index([5, 5, 2, 2, 9, 0, 1, 8, 8, 3])
    pandas_df = pandas.DataFrame({"a": range(10)}, index=index)
    batches = _collect(pd.DataFrame(pandas_df), 3)
    assert [batch.index.tolist() for batch in batches] == [
        [5, 5, 2],
        [2, 9, 0],
        [1, 8, 8],
        [3],
    ]
    tm.assert_frame_equal(pandas.concat(batches), pandas_df)


def test_categorical_dtype_preserved():
    pandas_df = pandas.DataFrame(
        {
            "cat": pandas.Categorical(
                ["a", "b", "c", "a", "c", "b", "a", "c"],
                categories=["c", "a", "b"],
            ),
            "num": range(8),
        }
    )
    modin_df = pd.DataFrame(pandas_df)

    batches = _collect(modin_df, 3)
    full = modin_df.modin.to_pandas()
    for batch in batches:
        # every batch carries the same (global) categorical dtype as to_pandas
        assert batch["cat"].dtype == full["cat"].dtype
        assert batch["cat"].cat.categories.tolist() == ["c", "a", "b"]
    tm.assert_frame_equal(pandas.concat(batches), full)


def test_categorical_union_across_partitions():
    # Build two frames whose partition-local categorical categories differ,
    # concatenate them in Modin and check the batches still match to_pandas.
    df1 = pandas.DataFrame(
        {"cat": pandas.Categorical(["a", "b", "a"], categories=["a", "b"])}
    )
    df2 = pandas.DataFrame(
        {"cat": pandas.Categorical(["b", "c", "c"], categories=["b", "c"])}
    )
    modin_df = pd.concat([pd.DataFrame(df1), pd.DataFrame(df2)], ignore_index=True)

    batches = _collect(modin_df, 2)
    full = modin_df.modin.to_pandas()
    tm.assert_frame_equal(pandas.concat(batches), full)


def test_empty_frame_yields_single_empty_typed_batch():
    pandas_df = pandas.DataFrame(
        {
            "a": pandas.Series(dtype="int64"),
            "b": pandas.Series(dtype="float64"),
            "c": pandas.Series(
                pandas.Categorical([], categories=["a", "b"]), dtype="category"
            ),
        }
    )
    modin_df = pd.DataFrame(pandas_df)

    batches = _collect(modin_df, 10)

    assert len(batches) == 1
    empty = batches[0]
    assert empty.empty
    assert empty.columns.tolist() == ["a", "b", "c"]
    assert empty["a"].dtype == np.dtype("int64")
    assert empty["b"].dtype == np.dtype("float64")
    tm.assert_frame_equal(empty, modin_df.modin.to_pandas())


def test_empty_frame_without_columns():
    modin_df = pd.DataFrame(columns=[])
    batches = _collect(modin_df, 5)
    assert len(batches) == 1
    assert batches[0].empty
    assert len(batches[0].columns) == 0


def test_columnless_frame_with_rows():
    pandas_df = pandas.DataFrame(index=pandas.Index([10, 20, 30, 40]))
    modin_df = pd.DataFrame(pandas_df)

    batches = _collect(modin_df, 2)
    assert [len(batch) for batch in batches] == [2, 2]
    tm.assert_frame_equal(pandas.concat(batches), modin_df.modin.to_pandas())


@pytest.mark.parametrize("bad_batch_size", [0, -1, -100, 1.5, "3", None, True, False])
def test_invalid_batch_size_rejected_eagerly(bad_batch_size):
    modin_df = pd.DataFrame({"a": [1, 2, 3]})
    # error is raised when the iterator is created, before any data is consumed
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        modin_df.modin.to_pandas_batches(batch_size=bad_batch_size)


def test_iterator_protocol():
    modin_df = pd.DataFrame({"a": range(10)})
    iterator = modin_df.modin.to_pandas_batches(batch_size=4)
    assert iter(iterator) is iterator
    batches = list(iterator)
    assert [len(batch) for batch in batches] == [4, 4, 2]
    with pytest.raises(StopIteration):
        next(iterator)


def test_mutating_batch_does_not_change_modin_frame():
    pandas_df = pandas.DataFrame(
        {"a": np.arange(12, dtype="int64"), "b": [f"s{i}" for i in range(12)]}
    )
    modin_df = pd.DataFrame(pandas_df)
    original = modin_df.modin.to_pandas()

    for batch in modin_df.modin.to_pandas_batches(batch_size=4):
        batch["a"] = -1
        batch["b"] = "mutated"
        batch.iloc[0, 0] = 999
        batch.index = ["x"] * len(batch)

    tm.assert_frame_equal(modin_df.modin.to_pandas(), original)
    tm.assert_frame_equal(modin_df.modin.to_pandas(), pandas_df)


def test_wide_frame_with_multiple_column_partitions():
    # 40 columns with NPartitions=4 (min column partition size 32) ->
    # 2 column partitions; include a categorical column in the second block.
    data = {
        f"col{i}": np.arange(37, dtype="int64") + i for i in range(39)
    }
    data["cat"] = pandas.Categorical(
        np.array(["x", "y", "z"])[np.arange(37) % 3],
        categories=["z", "y", "x"],
    )
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(pandas_df)
    frame = modin_df._query_compiler._modin_frame

    assert frame._partitions.shape[1] >= 2

    batches = _collect(modin_df, 10)
    full = modin_df.modin.to_pandas()
    assert list(batches[0].columns) == list(pandas_df.columns)
    for batch in batches:
        assert list(batch.dtypes) == list(full.dtypes)
    tm.assert_frame_equal(pandas.concat(batches), full)


def test_multiindex_axes_preserved():
    pandas_df = pandas.DataFrame(
        np.arange(24).reshape(6, 4),
        index=pandas.MultiIndex.from_product([["a", "b"], range(3)]),
        columns=pandas.MultiIndex.from_product([["x", "y"], [1, 2]]),
    )
    modin_df = pd.DataFrame(pandas_df)

    batches = _collect(modin_df, 4)
    assert all(batch.columns.equals(pandas_df.columns) for batch in batches)
    tm.assert_frame_equal(pandas.concat(batches), modin_df.modin.to_pandas())
    tm.assert_index_equal(
        pandas.concat(batches).index, pandas_df.index
    )


def test_numpy_integer_batch_size_accepted():
    modin_df = pd.DataFrame({"a": range(10)})
    batches = list(
        modin_df.modin.to_pandas_batches(batch_size=np.int64(4))
    )
    assert [len(batch) for batch in batches] == [4, 4, 2]


def test_series_not_supported():
    series = pd.Series([1, 2, 3])
    with pytest.raises(NotImplementedError):
        series.modin.to_pandas_batches(batch_size=2)


def test_multiple_independent_iterators():
    pandas_df = pandas.DataFrame({"a": range(10)})
    modin_df = pd.DataFrame(pandas_df)
    first = _collect(modin_df, 3)
    second = _collect(modin_df, 4)
    tm.assert_frame_equal(pandas.concat(first), pandas_df)
    tm.assert_frame_equal(pandas.concat(second), pandas_df)


def test_break_in_for_loop():
    modin_df = pd.DataFrame({"a": range(40)})
    seen = []
    for batch in modin_df.modin.to_pandas_batches(batch_size=10):
        seen.append(batch)
        if len(seen) == 2:
            break
    assert [len(batch) for batch in seen] == [10, 10]
    # the source frame stays fully usable after abandoning the iterator
    assert modin_df.modin.to_pandas()["a"].tolist() == list(range(40))


def test_creation_does_not_materialize_and_early_close_stops_pulling(monkeypatch):
    # 400 rows with NPartitions=4 (and default min partition size 32) ->
    # 4 row partitions of 100 rows each
    n_rows = 400
    pandas_df = pandas.DataFrame({"a": range(n_rows)})
    modin_df = pd.DataFrame(pandas_df)
    frame = modin_df._query_compiler._modin_frame
    mgr_cls = frame._partition_mgr_cls

    assert frame._partitions.shape[0] == 4

    requested_rows = []
    original_to_pandas = mgr_cls.to_pandas

    def spy_to_pandas(partitions):
        requested_rows.append(partitions.shape[0])
        return original_to_pandas(partitions)

    monkeypatch.setattr(mgr_cls, "to_pandas", spy_to_pandas)

    batch_size = 30
    iterator = modin_df.modin.to_pandas_batches(batch_size=batch_size)
    # merely creating the iterator pulls no partition data at all
    assert requested_rows == []

    first_batch = next(iterator)
    assert len(first_batch) == batch_size
    # only the first row of partitions was materialized
    assert requested_rows == [1]

    iterator.close()
    assert requested_rows == [1]

    # fully consuming a new iterator materializes each row partition exactly once
    full_iterator = modin_df.modin.to_pandas_batches(batch_size=batch_size)
    batches = list(full_iterator)
    assert requested_rows == [1] + [1] * frame._partitions.shape[0]
    n_full, remainder = divmod(n_rows, batch_size)
    expected_sizes = [batch_size] * n_full + (
        [remainder] if remainder else []
    )
    assert [len(batch) for batch in batches] == expected_sizes
    tm.assert_frame_equal(pandas.concat(batches), pandas_df)
