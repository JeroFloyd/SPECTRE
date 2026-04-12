from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"order_id", "customer_id", "product", "quantity", "unit_price", "status", "order_date"}
VALID_STATUSES = {"completed", "pending", "cancelled", "refunded", "processing"}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y"]


@dataclass
class ValidationReport:
    total_rows: int = 0
    missing_required: dict = field(default_factory=dict)
    type_errors: dict = field(default_factory=dict)
    invalid_enum_values: int = 0
    duplicate_rows: int = 0
    negative_values: int = 0
    invalid_dates: int = 0
    rows_flagged: int = 0
    rows_clean: int = 0
    passed: bool = False
    quality_score: float = 0.0

    def to_dict(self):
        return {
            "total_rows": self.total_rows,
            "missing_required": self.missing_required,
            "type_errors": self.type_errors,
            "invalid_enum_values": self.invalid_enum_values,
            "duplicate_rows": self.duplicate_rows,
            "negative_values": self.negative_values,
            "invalid_dates": self.invalid_dates,
            "rows_flagged": self.rows_flagged,
            "rows_clean": self.rows_clean,
            "passed": self.passed,
            "quality_score": round(self.quality_score, 4),
        }


@dataclass
class AggregateReport:
    total_batches: int = 0
    total_rows: int = 0
    total_revenue: float = 0.0
    avg_order_value: float = 0.0
    top_product: str = ""
    top_region: str = ""
    completed_pct: float = 0.0
    revenue_by_month: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "total_batches": self.total_batches,
            "total_rows": self.total_rows,
            "total_revenue": round(self.total_revenue, 2),
            "avg_order_value": round(self.avg_order_value, 2),
            "top_product": self.top_product,
            "top_region": self.top_region,
            "completed_pct": round(self.completed_pct, 4),
            "revenue_by_month": self.revenue_by_month,
        }


@dataclass
class RepairReport:
    """Tracks what transform() repaired vs dropped."""
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped: int = 0
    quantities_repaired: int = 0
    prices_repaired: int = 0
    statuses_repaired: int = 0
    dates_repaired: int = 0
    ids_deduplicated: int = 0

    def to_dict(self):
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
            "quantities_repaired": self.quantities_repaired,
            "prices_repaired": self.prices_repaired,
            "statuses_repaired": self.statuses_repaired,
            "dates_repaired": self.dates_repaired,
            "ids_deduplicated": self.ids_deduplicated,
            "survival_rate": round(self.rows_out / max(self.rows_in, 1), 4),
        }


@dataclass
class PipelineState:
    seed: int
    task: str
    data_dir: Path
    batch_file: str 

    df: pd.DataFrame | None = None
    source_file: str = ""
    rows_loaded: int = 0
    total_rows_loaded: int = 0
    columns: list[str] = field(default_factory=list)
    schema_hash: str = ""
    parse_count: int = 0

    validation_report: ValidationReport | None = None
    validate_count: int = 0

    repair_report: RepairReport | None = None
    all_transformed: list = field(default_factory=list)
    rows_after_transform: int = 0
    transform_count: int = 0
    revenue_total: float = 0.0
    derived_columns: list[str] = field(default_factory=list)

    aggregate_report: AggregateReport | None = None
    aggregate_count: int = 0

    output_path: str = ""
    rows_exported: int = 0
    export_count: int = 0
    quality_score: float = 0.0
    output_hash: str = ""

    def summary(self):
        vr = self.validation_report.to_dict() if self.validation_report else {}
        ar = self.aggregate_report.to_dict() if self.aggregate_report else {}
        rr = self.repair_report.to_dict() if self.repair_report else {}
        return {
            "source_file": self.source_file,
            "rows_loaded": self.rows_loaded,
            "total_rows_loaded": self.total_rows_loaded,
            "columns": self.columns,
            "schema_hash": self.schema_hash,
            "validation": vr,
            "repair": rr,
            "rows_after_transform": self.rows_after_transform,
            "revenue_total": round(self.revenue_total, 2),
            "derived_columns": self.derived_columns,
            "aggregate": ar,
            "output_path": self.output_path,
            "rows_exported": self.rows_exported,
            "quality_score": round(self.quality_score, 4),
            "output_hash": self.output_hash,
            "parse_count": self.parse_count,
            "validate_count": self.validate_count,
            "transform_count": self.transform_count,
            "aggregate_count": self.aggregate_count,
            "export_count": self.export_count,
        }


def _parse_date(val) -> str | None:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(str(val).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _is_valid_date(val) -> bool:
    return _parse_date(str(val)) is not None


def parse(ps: PipelineState):
    """Load selected batch first, then rotate through subsequent batches for multi-batch tasks."""
    
    if not hasattr(ps, '_batch_index'):
        ps._batch_index = 0
        import re
        match = re.search(r'orders_(\d+)\.csv', ps.batch_file)
        ps._start_batch = int(match.group(1)) if match else 1
    
   
    current_batch_num = ((ps._start_batch - 1 + ps._batch_index) % 6) + 1
    batch_filename = f"orders_{current_batch_num}.csv"
    batch_path = ps.data_dir / "raw" / batch_filename
    
    ps._batch_index += 1
    
    if not batch_path.exists():
        return f"Batch file {batch_filename} not found"
    
    try:
        try:
            df = pd.read_csv(batch_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(batch_path, encoding="latin-1")

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        ps.df = df
        ps.source_file = batch_filename
        ps.rows_loaded = len(df)
        ps.total_rows_loaded += len(df)
        ps.columns = list(df.columns)
        ps.schema_hash = hashlib.md5(",".join(ps.columns).encode()).hexdigest()[:8]
        ps.parse_count += 1
        ps.validation_report = None
        ps.repair_report = None
        ps.derived_columns = []
        logger.info("parse source=%s rows=%d cols=%d", batch_filename, ps.rows_loaded, len(ps.columns))
        return None
    except Exception as exc:
        return f"parse_data failed: {exc}"


def validate(ps: PipelineState):
    if ps.df is None:
        return "No data loaded. Run parse_data first."

    df = ps.df
    vr = ValidationReport(total_rows=len(df))

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        return f"validate_data failed: missing required columns {missing_cols}"

    for col in ["order_id", "customer_id", "product", "status", "order_date"]:
        n = int(df[col].isnull().sum() + (df[col].astype(str).str.strip() == "").sum())
        if n:
            vr.missing_required[col] = n

    for col in ["quantity", "unit_price"]:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_bad = max(int(coerced.isna().sum() - df[col].isna().sum()), 0)
        if n_bad:
            vr.type_errors[col] = n_bad

    vr.invalid_enum_values = int((~df["status"].isin(VALID_STATUSES)).sum())
    vr.duplicate_rows = int(df.duplicated(subset=["order_id"]).sum())

    qty_n = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    price_n = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
    vr.negative_values = int(((qty_n <= 0) | (price_n <= 0)).sum())

    vr.invalid_dates = sum(
        1 for v in df["order_date"].astype(str)
        if not _is_valid_date(v)
    )

    flagged = (
        df["order_id"].isnull() |
        (df["order_id"].astype(str).str.strip() == "") |
        pd.to_numeric(df["quantity"], errors="coerce").isna() |
        pd.to_numeric(df["unit_price"], errors="coerce").isna() |
        ~df["status"].isin(VALID_STATUSES)
    )
    vr.rows_flagged = int(flagged.sum())
    vr.rows_clean = vr.total_rows - vr.rows_flagged

    total_issues = (
        sum(vr.missing_required.values()) +
        sum(vr.type_errors.values()) +
        vr.invalid_enum_values +
        vr.duplicate_rows +
        vr.negative_values +
        vr.invalid_dates
    )
    vr.quality_score = max(0.0, 1.0 - (total_issues / max(vr.total_rows, 1)))
    vr.passed = vr.quality_score >= 0.70
    ps.validation_report = vr
    ps.validate_count += 1
    ps.quality_score = round(vr.quality_score, 4)

    logger.info("validate rows=%d flagged=%d quality=%.3f passed=%s",
                vr.total_rows, vr.rows_flagged, vr.quality_score, vr.passed)
    return None

def transform(ps: PipelineState):
    """
    AGENT RECONSTRUCTION — repair corrupted fields rather than dropping rows.
    This is the CORE intelligence of the agent.
    """
    if ps.df is None:
        return "No data loaded. Run parse_data first."
    if ps.validation_report is None:
        return "Data not validated. Run validate_data before transform_data."

    df = ps.df.copy()
    rr = RepairReport(rows_in=len(df))

    col_map = {"qty": "quantity", "price": "unit_price", "unitprice": "unit_price"}
    df = df.rename(columns=lambda c: col_map.get(c, c))

    # Drop only unsalvageable rows
    before = len(df)
    df = df[df["order_id"].astype(str).str.strip() != ""]
    df = df[df["customer_id"].astype(str).str.strip() != ""]
    df = df.dropna(subset=["order_id", "customer_id"])
    rr.rows_dropped = before - len(df)

    # Repair duplicate order_ids
    seen: dict = {}
    new_ids = []
    for oid in df["order_id"].astype(str):
        if oid in seen:
            seen[oid] += 1
            new_ids.append(f"{oid}_r{seen[oid]}")
            rr.ids_deduplicated += 1
        else:
            seen[oid] = 0
            new_ids.append(oid)
    df["order_id"] = new_ids

    # Repair quantity
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    qty_bad = df["quantity"].isna().sum() + (df["quantity"].fillna(0) < 1).sum()
    df["quantity"] = df["quantity"].fillna(1).clip(lower=1).astype(int)
    rr.quantities_repaired = int(qty_bad)

    # Repair unit_price
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    valid_prices = df["unit_price"].dropna()
    valid_prices = valid_prices[valid_prices > 0]
    median_price = float(valid_prices.median()) if len(valid_prices) > 0 else 9.99
    price_bad = df["unit_price"].isna().sum() + (df["unit_price"].fillna(0) < 0).sum()
    df["unit_price"] = df["unit_price"].abs().fillna(median_price).round(2)
    df.loc[df["unit_price"] == 0, "unit_price"] = median_price
    rr.prices_repaired = int(price_bad)

    # Repair status
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    bad_status_mask = ~df["status"].isin(VALID_STATUSES)
    rr.statuses_repaired = int(bad_status_mask.sum())
    df.loc[bad_status_mask, "status"] = "pending"

    # Repair order_date
    parsed = df["order_date"].apply(lambda v: _parse_date(str(v)))
    bad_date_mask = parsed.isna()
    rr.dates_repaired = int(bad_date_mask.sum())
    valid_dates = parsed.dropna()
    fallback_date = str(valid_dates.mode().iloc[0]) if len(valid_dates) > 0 else "2024-01-01"
    df["order_date"] = parsed.fillna(fallback_date)

    # Add derived columns
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    df["order_year"] = df["order_date"].str[:4].astype(int)
    df["order_month"] = df["order_date"].str[5:7].astype(int)
    df["order_quarter"] = ((df["order_month"] - 1) // 3 + 1).astype(int)
    df["price_tier"] = pd.cut(
        df["unit_price"],
        bins=[0, 50, 200, 500, float("inf")],
        labels=["budget", "mid", "premium", "luxury"]
    ).astype(str)

    rr.rows_out = len(df)

    batch_revenue = float(df["revenue"].sum())
    ps.df = df
    ps.all_transformed.append(df)
    ps.rows_after_transform = sum(len(d) for d in ps.all_transformed)
    ps.revenue_total += batch_revenue
    ps.derived_columns = ["revenue", "order_year", "order_month", "order_quarter", "price_tier"]
    ps.repair_report = rr
    ps.transform_count += 1

    total_repaired = (rr.quantities_repaired + rr.prices_repaired +
                      rr.statuses_repaired + rr.dates_repaired + rr.ids_deduplicated)
    repair_penalty = total_repaired / max(rr.rows_in, 1) 
    survival_rate = rr.rows_out / max(rr.rows_in, 1)
    
    base_quality = 0.95 - (repair_penalty * 0.30)
    ps.quality_score = round(max(0.70, min(0.97, base_quality * survival_rate)), 4)

    total_repaired = (rr.quantities_repaired + rr.prices_repaired +

                      rr.statuses_repaired + rr.dates_repaired + rr.ids_deduplicated)
    logger.info(
        "transform rows_in=%d rows_out=%d dropped=%d repaired=%d revenue=%.2f",
        rr.rows_in, rr.rows_out, rr.rows_dropped, total_repaired, batch_revenue
    )
    return None


def aggregate(ps: PipelineState):
    if not ps.all_transformed:
        return "No transformed data available. Run transform_data first."

    combined = pd.concat(ps.all_transformed, ignore_index=True)
    ar = AggregateReport()
    ar.total_batches = ps.transform_count
    ar.total_rows = len(combined)
    ar.total_revenue = float(combined["revenue"].sum())
    ar.avg_order_value = float(combined["revenue"].mean()) if len(combined) else 0.0

    if "product" in combined.columns:
        ar.top_product = str(combined.groupby("product")["revenue"].sum().idxmax())
    if "region" in combined.columns:
        ar.top_region = str(combined.groupby("region")["revenue"].sum().idxmax())
    if "status" in combined.columns:
        ar.completed_pct = round(
            (combined["status"] == "completed").sum() / max(len(combined), 1), 4
        )
    if "order_month" in combined.columns:
        combined["ym"] = (
            combined["order_year"].astype(str) + "-" +
            combined["order_month"].astype(str).str.zfill(2)
        )
        ar.revenue_by_month = {
            k: float(v)
            for k, v in combined.groupby("ym")["revenue"].sum().round(2).items()
        }

    ps.aggregate_report = ar
    ps.aggregate_count += 1
    logger.info("aggregate batches=%d total_rows=%d revenue=%.2f top_product=%s",
                ar.total_batches, ar.total_rows, ar.total_revenue, ar.top_product)
    return None


def export(ps: PipelineState):
    if not ps.all_transformed:
        return "No transformed data to export. Run transform_data first."

    combined = pd.concat(ps.all_transformed, ignore_index=True)
    if len(combined) == 0:
        return "DataFrame is empty after reconstruction. Nothing to export."

    processed_dir = ps.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_name = f"orders_processed_{timestamp}.csv"
    output_path = processed_dir / output_name

    try:
        combined.to_csv(output_path, index=False)
    except Exception as exc:
        return f"export_result failed: {exc}"

    content = output_path.read_bytes()
    ps.output_hash = hashlib.sha256(content).hexdigest()[:16]
    ps.output_path = str(output_path)
    ps.rows_exported = len(combined)

    null_rate = combined.isnull().sum().sum() / max(combined.size, 1)
    cleanliness = 1.0 - null_rate
    completeness = min(1.0, ps.rows_exported / max(ps.total_rows_loaded, 1))
    derived_bonus = 0.05 if ps.derived_columns else 0.0
    agg_bonus = 0.05 if ps.aggregate_report else 0.0
    ps.quality_score = min(1.0, 0.55 * cleanliness + 0.35 * completeness + derived_bonus + agg_bonus)
    ps.export_count += 1
    logger.info("export path=%s rows=%d quality=%.3f hash=%s",
                output_name, ps.rows_exported, ps.quality_score, ps.output_hash)
    return None



