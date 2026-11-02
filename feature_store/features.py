"""
features.py
───────────
Feast Feature Definitions for Sentinel Fraud Detection.
Defines entity keys and historical batch feature views for sub-millisecond online retrieval.
"""

from datetime import timedelta
import os
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float32, Int64

# Current file directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_SOURCE_PATH = os.path.join(CURRENT_DIR, "data", "card_risk_features.parquet")

# ─── Entity ───────────────────────────────────────────────────────────────────
card_entity = Entity(
    name="card1",
    join_keys=["card1"],
    description="Primary payment card token / identifier",
)

# ─── Source ───────────────────────────────────────────────────────────────────
card_source = FileSource(
    name="card_risk_source",
    path=PARQUET_SOURCE_PATH,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ─── Feature View ─────────────────────────────────────────────────────────────
card_risk_fv = FeatureView(
    name="card_risk_profile",
    entities=[card_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="card1_TransactionAmt_mean", dtype=Float32),
        Field(name="card1_TransactionAmt_std", dtype=Float32),
        Field(name="card1_freq", dtype=Int64),
    ],
    online=True,
    source=card_source,
    description="Historical payment behavior and transaction frequency statistics per card token",
)
