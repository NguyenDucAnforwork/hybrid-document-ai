"""Pydantic output contract — the API/JSON schema from PLAN.md §8."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class FieldValue(BaseModel):
    value: Optional[str | int | float] = None
    confidence: float = 0.0


class QualityReport(BaseModel):
    blur_score: float
    is_blurry: bool
    is_dark: bool
    low_resolution: bool
    is_rotated: bool
    quality_pass: bool
    issues: list[str] = Field(default_factory=list)
    action: Optional[str] = None


class DocumentResult(BaseModel):
    document_id: str
    document_type: str = "receipt"
    route: str = "traditional_ocr"          # traditional_ocr | vlm_fallback
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    line_items: list[dict] = Field(default_factory=list)  # transaction rows (bank_statement)
    quality: QualityReport
    needs_human_review: bool = False
    model_versions: dict[str, str] = Field(default_factory=dict)  # audit/traceability
    error: Optional[str] = None


class BatchSummary(BaseModel):
    total: int = 0
    success: int = 0
    needs_review: int = 0
    failed: int = 0
    vlm_fallback: int = 0


class BatchJob(BaseModel):
    job_id: str
    status: str = "queued"   # queued|processing|partial_completed|completed|failed
    total_documents: int = 0
    summary: BatchSummary = Field(default_factory=BatchSummary)
