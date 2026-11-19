# SPDX-License-Identifier: MIT
# Based on the GATE implementation by dm4ml.
# Copyright (c) 2023 dm4ml
# Modified for AdaGATE, 2026.
# See the LICENSE file for details.

import typing

import pandas as pd
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

from .summary import Summary


def compute_embeddings(column: pd.Series, column_type: str) -> pd.Series:
    assert column_type in [
        "text",
        "image",
    ], "column_type must be text or image"

    model = SentenceTransformer("clip-ViT-B-32")

    def compute_embedding_helper(text: str) -> typing.List[float]:
        if column_type == "image":
            try:
                img = Image.open(text)
                return model.encode(img)
            except FileNotFoundError:
                img = Image.open(requests.get(text, stream=True).raw)
                return model.encode(img)
            except Exception as e:
                raise e

        return model.encode(text)

    return column.apply(compute_embedding_helper)


# Public entry point for computing partition-level summaries.
def summarize(
    df: pd.DataFrame,
    columns: typing.List[str] = [],
    embedding_column_map: typing.Dict[str, str] = {},
    partition_key: str = "",
    previous_summaries: typing.List[Summary] = [],
) -> typing.List[Summary]:

    if partition_key == "group":
        raise ValueError("Please rename the partition_key; it cannot be `group`.")

    if len(previous_summaries) == 0:
        if len(columns) == 0 and len(embedding_column_map) == 0:
            raise ValueError(
                "You must pass in some columns if you do not have any previous"
                " summaries."
            )
        if not partition_key:
            raise ValueError(
                "You must pass in a partition column if you do not have any"
                " previous summaries."
            )

    summary = Summary.fromRaw(
        df,
        columns=columns,
        embedding_column_map=embedding_column_map,
        partition_key=partition_key,
        previous_summaries=previous_summaries,
    )

    return summary
