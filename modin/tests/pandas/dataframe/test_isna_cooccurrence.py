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
from modin.config import NPartitions
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import df_equals

NPartitions.put(4)


@pytest.fixture
def npartitions(request):
    """Temporarily set the number of partitions for a test."""
    old_value = NPartitions.get()
    NPartitions.put(request.param)
    yield
    NPartitions.put(old_value)


def reference_cooccurrence(df, columns):
    """Build the expected matrix with plain pandas ``isna`` semantics."""
    mask = df[columns].isna().to_numpy(dtype=np.int64)
    matrix = mask.T @ mask
    labels = pandas.Index(columns)
    return pandas.DataFrame(matrix, index=labels, columns=labels)


def make_multi_partition_df():
    nrows = 37
    data = {
        "a": [i if i % 2 == 0 else np.nan for i in range(nrows)],
        "b": [i if i % 3 != 0 else np.nan for i in range(nrows)],
        "c": [i if i % 5 != 0 else np.nan for i in range(nrows)],
        "d": [np.nan] * 5 + list(range(nrows - 5)),
        "e": list(range(nrows)),
    }
    return pandas.DataFrame(data)


def test_basic_cooccurrence_and_diagonal():
    pandas_df = make_multi_partition_df()
    modin_df = pd.DataFrame(pandas_df)
    columns = ["a", "b", "c", "d"]

    # The frame must span several row partitions, otherwise this test does not
    # exercise the local-count + merge code path.
    assert modin_df._query_compiler._modin_frame._partitions.shape[0] > 1

    result = modin_df.isna_cooccurrence(columns)

    pandas.testing.assert_frame_equal(
        result, reference_cooccurrence(pandas_df, columns)
    )
    # The result is symmetric and its diagonal matches per-column isna counts.
    numpy_result = result.to_numpy()
    np.testing.assert_array_equal(numpy_result, numpy_result.T)
    np.testing.assert_array_equal(
        np.diag(numpy_result),
        pandas_df[columns].isna().sum().to_numpy(),
    )
    assert result.index.equals(pandas.Index(columns))
    assert result.columns.equals(pandas.Index(columns))
    assert np.issubdtype(result.values.dtype, np.integer)


@pytest.mark.parametrize("npartitions", [1, 3, 4, 7], indirect=True)
def test_cooccurrence_for_various_partition_counts(npartitions):
    pandas_df = make_multi_partition_df()
    columns = ["e", "a", "d", "b"]
    result = pd.DataFrame(pandas_df).isna_cooccurrence(columns)
    pandas.testing.assert_frame_equal(
        result, reference_cooccurrence(pandas_df, columns)
    )


def test_pairs_live_in_different_partitions():
    # Missing pairs are placed into different row partitions on purpose, so a
    # correct answer requires merging local counts of all partitions.
    nrows = 40
    pandas_df = pandas.DataFrame(
        {
            "a": [np.nan] * 10 + list(range(nrows - 10)),
            "b": list(range(nrows // 2))
            + [np.nan] * (nrows // 2),
        }
    )
    modin_df = pd.DataFrame(pandas_df)
    result = modin_df.isna_cooccurrence(["a", "b"])
    expected = reference_cooccurrence(pandas_df, ["a", "b"])
    pandas.testing.assert_frame_equal(result, expected)
    # 'a' is missing in the first half, 'b' in the second half -> never together.
    assert result.loc["a", "b"] == 0
    assert result.loc["a", "a"] == 10
    assert result.loc["b", "b"] == 20


def test_different_missing_value_representations():
    pandas_df = pandas.DataFrame(
        {
            "float_col": [1.0, np.nan, 3.0, float("nan"), 5.0],
            "object_col": ["x", None, "z", None, "w"],
            # empty string is NOT a missing value
            "string_col": ["s", "", pd.NA, "t", pd.NA],
            "nullable_int": pandas.Series(
                [1, pd.NA, 3, 4, pd.NA], dtype="Int64"
            ),
            "datetime_col": pandas.Series(
                [pandas.Timestamp("2020-01-01"), pandas.NaT, pandas.NaT,
                 pandas.Timestamp("2020-01-04"), pandas.NaT],
                dtype="datetime64[ns]",
            ),
        }
    )
    columns = ["float_col", "object_col", "string_col",
               "nullable_int", "datetime_col"]
    result = pd.DataFrame(pandas_df).isna_cooccurrence(columns)
    pandas.testing.assert_frame_equal(
        result, reference_cooccurrence(pandas_df, columns)
    )


def test_result_follows_requested_column_order():
    pandas_df = make_multi_partition_df()
    requested = ["d", "a", "e", "b"]
    result = pd.DataFrame(pandas_df).isna_cooccurrence(requested)
    assert result.index.tolist() == requested
    assert result.columns.tolist() == requested
    pandas.testing.assert_frame_equal(
        result, reference_cooccurrence(pandas_df, requested)
    )


def test_single_column():
    pandas_df = make_multi_partition_df()
    result = pd.DataFrame(pandas_df).isna_cooccurrence(["b"])
    assert result.shape == (1, 1)
    assert result.index.tolist() == ["b"]
    assert result.iloc[0, 0] == int(pandas_df["b"].isna().sum())


def test_empty_input_returns_empty_square_matrix():
    modin_df = pd.DataFrame(make_multi_partition_df())
    result = modin_df.isna_cooccurrence([])
    assert isinstance(result, pandas.DataFrame)
    assert result.shape == (0, 0)


def test_empty_frame_gives_zero_matrix():
    pandas_df = pandas.DataFrame(
        {
            "a": pandas.Series(dtype="float64"),
            "b": pandas.Series(dtype="object"),
            "c": pandas.Series(dtype="datetime64[ns]"),
        }
    )
    modin_df = pd.DataFrame(pandas_df)
    result = modin_df.isna_cooccurrence(["a", "b", "c"])
    assert result.shape == (3, 3)
    assert result.index.tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(result.to_numpy(), np.zeros((3, 3), dtype=np.int64))


def test_duplicate_columns_in_input_raise():
    modin_df = pd.DataFrame(make_multi_partition_df())
    with pytest.raises(ValueError, match="duplicate"):
        modin_df.isna_cooccurrence(["a", "b", "a"])


def test_missing_column_raises():
    modin_df = pd.DataFrame(make_multi_partition_df())
    with pytest.raises(KeyError, match="does not exist"):
        modin_df.isna_cooccurrence(["a", "missing"])


def test_non_unique_column_label_in_frame_raises():
    pandas_df = pandas.DataFrame(
        [[0, 1, 2], [3, 4, 5]], columns=["a", "b", "a"]
    )
    modin_df = pd.DataFrame(pandas_df)
    with pytest.raises(ValueError, match="not unique"):
        modin_df.isna_cooccurrence(["a"])
    # A non-requested duplicated label must not affect other columns.
    result = modin_df.isna_cooccurrence(["b"])
    pandas.testing.assert_frame_equal(
        result, reference_cooccurrence(pandas_df, ["b"])
    )


def test_non_sequence_input_raises():
    modin_df = pd.DataFrame(make_multi_partition_df())
    with pytest.raises(TypeError):
        modin_df.isna_cooccurrence("a")
    with pytest.raises(TypeError):
        modin_df.isna_cooccurrence(None)


def test_original_frame_and_row_order_unchanged():
    pandas_df = make_multi_partition_df()
    modin_df = pd.DataFrame(pandas_df)
    original = to_pandas(modin_df).copy()

    modin_df.isna_cooccurrence(["c", "a", "d"])

    df_equals(modin_df, pandas_df)
    pandas.testing.assert_frame_equal(to_pandas(modin_df), original)
    assert modin_df.columns.tolist() == pandas_df.columns.tolist()
    assert modin_df.index.tolist() == pandas_df.index.tolist()


def test_columns_spanning_multiple_column_partitions():
    ncols = 8
    nrows = 25
    data = {
        f"col{i}": [
            np.nan if (row + i) % 3 == 0 else row for row in range(nrows)
        ]
        for i in range(ncols)
    }
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(pandas_df)
    # Requested columns come from both ends of the column-partition grid.
    requested = [f"col{i}" for i in (7, 0, 3, 5, 1)]
    result = modin_df.isna_cooccurrence(requested)
    pandas.testing.assert_frame_equal(
        result, reference_cooccurrence(pandas_df, requested)
    )
