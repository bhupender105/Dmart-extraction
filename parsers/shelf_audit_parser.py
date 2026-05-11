"""
LangChain Pydantic-based JSON output parser for the Retail Shelf Audit API.

Uses langchain_core.output_parsers.JsonOutputParser with a Pydantic schema so
the model response is automatically validated and structured.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


# ── Schema ────────────────────────────────────────────────────────────────────

class AuditSummary(BaseModel):
    total_shelves_detected: int = Field(
        description="Total number of distinct shelves detected in the image."
    )
    abc_shelf_share_percentage: float = Field(
        description="Approximate percentage of shelf facings occupied by ABC Foods products."
    )
    is_planogram_compliant: bool = Field(
        description="Whether the current shelf arrangement matches the reference planogram."
    )
    overall_store_lighting_and_visibility: str = Field(
        description="Overall image quality / store lighting: Good, Fair, or Poor."
    )


class ProductDetected(BaseModel):
    brand_name: str = Field(description="Brand name of the product (e.g. 'ABC Foods', 'Kelloggs').")
    product_name: str = Field(description="Specific product name (e.g. 'ABC Crunchy Cornflakes').")
    packaging_type: str = Field(description="Type of packaging: Box, Pouch, Bag, Bottle, Sack, etc.")
    is_abc_product: bool = Field(description="True if the product belongs to the ABC Foods brand.")
    facings_count: int = Field(description="Number of front-facing units visible for this product.")
    units_to_order: int = Field(
        description="Units needed to reach planogram target (9 per ABC product): max(0, 9 - facings_count*3). Always 0 for non-ABC products."
    )
    is_out_of_stock: bool = Field(
        description="True if facings_count is 0 — product is completely missing from this shelf."
    )
    visible_price_tag: Optional[str] = Field(
        default=None,
        description="Price tag text if visible, otherwise null."
    )


class ShelfDetail(BaseModel):
    shelf_index: int = Field(
        description="1-based index of the shelf counting from the top (1 = top shelf)."
    )
    shelf_level_type: str = Field(
        description="Human-readable level: Top, Eye-level, Middle, or Bottom."
    )
    products_detected: List[ProductDetected] = Field(
        description="All products identified on this shelf."
    )


class ComplianceIssue(BaseModel):
    issue_type: str = Field(
        description="Category of the issue: Missing SKU, Wrong Placement, Low Stock, or Competitor Intrusion."
    )
    description: str = Field(
        description="Detailed description of the compliance issue."
    )


class RecommendedActionTask(BaseModel):
    task_type: str = Field(
        description="Type of corrective action: Restock Order, Merchandising Correction, or Competitor Flag."
    )
    urgency: str = Field(description="Priority level: High, Medium, or Low.")
    action_description: str = Field(
        description="Clear, actionable instruction for the field team."
    )


class ShelfAuditReport(BaseModel):
    audit_summary: AuditSummary = Field(
        description="High-level summary of the shelf audit."
    )
    shelf_details: List[ShelfDetail] = Field(
        description="Per-shelf breakdown of all products detected."
    )
    compliance_issues: List[ComplianceIssue] = Field(
        description="List of detected compliance/planogram issues."
    )
    recommended_action_tasks: List[RecommendedActionTask] = Field(
        description="Prioritised list of corrective tasks for the field team."
    )


# ── Parser instance ───────────────────────────────────────────────────────────

ShelfAuditParser = JsonOutputParser(pydantic_object=ShelfAuditReport)


def get_format_instructions() -> str:
    """Return the format instructions string to inject into the system prompt."""
    return ShelfAuditParser.get_format_instructions()
