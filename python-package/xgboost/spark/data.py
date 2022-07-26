"""Utilities for processing spark partitions."""
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from xgboost.compat import concat

from xgboost import DataIter, DeviceQuantileDMatrix, DMatrix


def stack_series(series: pd.Series) -> np.ndarray:
    """Stack a series of arrays."""
    array = series.to_numpy(copy=False)
    array = np.stack(array)
    return array


# Global constant for defining column alias shared between estimator and data
# processing procedures.
Alias = namedtuple("Alias", ("data", "label", "weight", "margin", "valid"))
alias = Alias("values", "label", "weight", "baseMargin", "validationIndicator")


def concat_or_none(seq: Optional[Sequence[np.ndarray]]) -> Optional[np.ndarray]:
    """Concatenate the data if it's not None."""
    if seq:
        return concat(seq)
    return None


def cache_partitions(
    iterator: Iterator[pd.DataFrame], append: Callable[[pd.DataFrame, str, bool], None]
) -> None:
    """Extract partitions from pyspark iterator. `append` is a user defined function for
    accepting new partition."""

    def make_blob(part: pd.DataFrame, is_valid: bool) -> None:
        append(part, alias.data, is_valid)
        append(part, alias.label, is_valid)
        append(part, alias.weight, is_valid)
        append(part, alias.margin, is_valid)

    has_validation: Optional[bool] = None

    for part in iterator:
        if has_validation is None:
            has_validation = alias.valid in part.columns
        if has_validation is True:
            assert alias.valid in part.columns

        if has_validation:
            train = part.loc[~part[alias.valid], :]
            valid = part.loc[part[alias.valid], :]
        else:
            train, valid = part, None

        make_blob(train, False)
        if valid is not None:
            make_blob(valid, True)


class PartIter(DataIter):
    """Iterator for creating Quantile DMatrix from partitions."""

    def __init__(self, data: Dict[str, List], on_device: bool) -> None:
        self._iter = 0
        self._cuda = on_device
        self._data = data

        super().__init__()

    def _fetch(self, data: Optional[Sequence[pd.DataFrame]]) -> Optional[pd.DataFrame]:
        if not data:
            return None

        if self._cuda:
            import cudf  # pylint: disable=import-error

            return cudf.DataFrame(data[self._iter])

        return data[self._iter]

    def next(self, input_data: Callable) -> int:
        if self._iter == len(self._data[alias.data]):
            return 0
        input_data(
            data=self._fetch(self._data[alias.data]),
            label=self._fetch(self._data.get(alias.label, None)),
            weight=self._fetch(self._data.get(alias.weight, None)),
            base_margin=self._fetch(self._data.get(alias.margin, None)),
        )
        self._iter += 1
        return 1

    def reset(self) -> None:
        self._iter = 0


def create_dmatrix_from_partitions(
    iterator: Iterator[pd.DataFrame],
    feature_cols: Optional[Sequence[str]],
    kwargs: Dict[str, Any],  # use dict to make sure this parameter is passed.
) -> Tuple[DMatrix, Optional[DMatrix]]:
    """Create DMatrix from spark data partitions. This is not particularly efficient as
    we need to convert the pandas series format to numpy then concatenate all the data.

    Parameters
    ----------
    iterator :
        Pyspark partition iterator.
    kwargs :
        Metainfo for DMatrix.

    """

    train_data: Dict[str, List[np.ndarray]] = defaultdict(list)
    valid_data: Dict[str, List[np.ndarray]] = defaultdict(list)

    n_features: int = 0

    def append_m(part: pd.DataFrame, name: str, is_valid: bool) -> None:
        nonlocal n_features
        if name in part.columns:
            array = part[name]
            if name == alias.data:
                array = stack_series(array)
                if n_features == 0:
                    n_features = array.shape[1]
                assert n_features == array.shape[1]

            if is_valid:
                valid_data[name].append(array)
            else:
                train_data[name].append(array)

    def append_dqm(part: pd.DataFrame, name: str, is_valid: bool) -> None:
        """Preprocessing for DeviceQuantileDMatrix"""
        nonlocal n_features
        if name == alias.data or name in part.columns:
            if name == alias.data:
                cname = feature_cols
            else:
                cname = name

            array = part[cname]
            if name == alias.data:
                if n_features == 0:
                    n_features = array.shape[1]
                assert n_features == array.shape[1]

            if is_valid:
                valid_data[name].append(array)
            else:
                train_data[name].append(array)

    def make(values: Dict[str, List[np.ndarray]], kwargs: Dict[str, Any]) -> DMatrix:
        data = concat_or_none(values[alias.data])
        label = concat_or_none(values.get(alias.label, None))
        weight = concat_or_none(values.get(alias.weight, None))
        margin = concat_or_none(values.get(alias.margin, None))
        return DMatrix(
            data=data, label=label, weight=weight, base_margin=margin, **kwargs
        )

    is_dmatrix = feature_cols is None
    if is_dmatrix:
        cache_partitions(iterator, append_m)
        dtrain = make(train_data, kwargs)
    else:
        cache_partitions(iterator, append_dqm)
        it = PartIter(train_data, True)
        dtrain = DeviceQuantileDMatrix(it, **kwargs)

    dvalid = make(valid_data, kwargs) if len(valid_data) != 0 else None

    assert dtrain.num_col() == n_features
    if dvalid is not None:
        assert dvalid.num_col() == dtrain.num_col()

    return dtrain, dvalid