from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from compact_rag.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedTable:
    page_number: int
    rows: int
    columns: int
    markdown: str
    quality_score: float = 0.0
    method: str = ""
    metadata: dict = field(default_factory=dict)


class TableExtractor:
    """Extracts tables from PDF, HTML, and DOCX files."""

    @staticmethod
    def evaluate_table_quality(markdown_table: str) -> dict:
        lines = [l.strip() for l in markdown_table.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return {"valid": False, "score": 0.0, "reason": "Too few lines"}

        has_separator = any(
            re.match(r"^\|[-| :]+\|$", line) for line in lines
        )
        if not has_separator:
            return {"valid": False, "score": 0.0, "reason": "No separator row"}

        data_lines = [l for l in lines if not re.match(r"^\|[-| :]+\|$", l)]
        if len(data_lines) < 1:
            return {"valid": False, "score": 0.0, "reason": "No data rows"}

        col_counts = [len(l.split("|")) - 2 for l in data_lines]
        consistent = len(set(col_counts)) == 1

        score = 0.5
        if len(data_lines) >= 2:
            score += 0.2
        if consistent:
            score += 0.3
        score = min(score, 1.0)

        return {
            "valid": score >= 0.5,
            "score": score,
            "rows": len(data_lines),
            "columns": col_counts[0] if col_counts else 0,
            "consistent_columns": consistent,
        }

    def extract_from_pdf(self, file_path: str) -> list[ExtractedTable]:
        tables: list[ExtractedTable] = []

        camelot_tables = self._extract_camelot(file_path)
        if camelot_tables:
            tables.extend(camelot_tables)
        else:
            logger.info("Camelot found no tables; falling back to pdfplumber", file=file_path)
            tables = self._extract_pdfplumber(file_path)

        return tables

    def _extract_camelot(self, file_path: str) -> list[ExtractedTable]:
        try:
            import camelot
        except ImportError:
            logger.warning("camelot-py not installed; skipping PDF table extraction via camelot")
            return []

        try:
            extracted = camelot.read_pdf(file_path, pages="all", flavor="lattice")
            if not extracted:
                extracted = camelot.read_pdf(file_path, pages="all", flavor="stream")
        except Exception as e:
            logger.warning("Camelot extraction failed", error=str(e))
            return []

        tables: list[ExtractedTable] = []
        for t in extracted:
            df = t.df
            rows = len(df)
            cols = len(df.columns)
            md = _dataframe_to_markdown(df)
            quality = self.evaluate_table_quality(md)
            tables.append(
                ExtractedTable(
                    page_number=t.page,
                    rows=rows,
                    columns=cols,
                    markdown=md,
                    quality_score=quality["score"],
                    method=f"camelot_{t.flavor}",
                    metadata={"accuracy": getattr(t, "accuracy", 0), "file": Path(file_path).name},
                )
            )

        logger.info(
            "Camelot tables extracted",
            file=Path(file_path).name,
            count=len(tables),
        )
        return tables

    def _extract_pdfplumber(self, file_path: str) -> list[ExtractedTable]:
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed; PDF table fallback unavailable")
            return []

        tables: list[ExtractedTable] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_tables()
                    if not extracted:
                        continue
                    for t in extracted:
                        if not t or len(t) < 2:
                            continue
                        md = _matrix_to_markdown(t)
                        quality = self.evaluate_table_quality(md)
                        if quality["valid"]:
                            tables.append(
                                ExtractedTable(
                                    page_number=page.page_number,
                                    rows=len(t),
                                    columns=len(t[0]) if t else 0,
                                    markdown=md,
                                    quality_score=quality["score"],
                                    method="pdfplumber",
                                    metadata={"file": Path(file_path).name},
                                )
                            )
        except Exception as e:
            logger.warning("pdfplumber extraction failed", error=str(e))

        logger.info(
            "pdfplumber tables extracted",
            file=Path(file_path).name,
            count=len(tables),
        )
        return tables

    def extract_from_html(self, html_content: str) -> list[ExtractedTable]:
        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify
        except ImportError:
            logger.warning("bs4/markdownify not installed; HTML table extraction unavailable")
            return []

        tables: list[ExtractedTable] = []
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            for i, table in enumerate(soup.find_all("table")):
                md = markdownify(str(table))
                quality = self.evaluate_table_quality(md)
                if quality["valid"]:
                    rows = len(table.find_all("tr"))
                    header_cells = table.find_all("th")
                    first_row_cells = table.find("tr")
                    cols = 0
                    if header_cells:
                        cols = len(header_cells)
                    elif first_row_cells:
                        cols = len(first_row_cells.find_all(["td", "th"]))

                    tables.append(
                        ExtractedTable(
                            page_number=i + 1,
                            rows=rows,
                            columns=cols,
                            markdown=md,
                            quality_score=quality["score"],
                            method="html",
                            metadata={"source": "html"},
                        )
                    )
        except Exception as e:
            logger.warning("HTML table extraction failed", error=str(e))

        return tables

    def extract_from_docx(self, file_path: str) -> list[ExtractedTable]:
        try:
            from docx import Document
            from compact_rag.ingestion.loader import _docx_table_to_markdown
        except ImportError:
            logger.warning("python-docx not installed; DOCX table extraction unavailable")
            return []

        tables: list[ExtractedTable] = []
        try:
            doc = Document(file_path)
            for i, table in enumerate(doc.tables):
                md = _docx_table_to_markdown(table)
                if not md:
                    continue
                quality = self.evaluate_table_quality(md)
                if quality["valid"]:
                    tables.append(
                        ExtractedTable(
                            page_number=i + 1,
                            rows=len(table.rows),
                            columns=len(table.columns),
                            markdown=md,
                            quality_score=quality["score"],
                            method="docx",
                            metadata={"file": Path(file_path).name},
                        )
                    )
        except Exception as e:
            logger.warning("DOCX table extraction failed", error=str(e))

        logger.info(
            "DOCX tables extracted",
            file=Path(file_path).name,
            count=len(tables),
        )
        return tables


def _dataframe_to_markdown(df) -> str:
    rows = [df.columns.tolist()] + df.values.tolist()
    return _matrix_to_markdown(rows)


def _matrix_to_markdown(rows: list[list]) -> str:
    if not rows:
        return ""
    cleaned: list[list[str]] = []
    for row in rows:
        cleaned.append([str(cell or "").replace("\n", " ").strip() for cell in row])
    col_count = max(len(r) for r in cleaned)
    for r in cleaned:
        while len(r) < col_count:
            r.append("")
    lines: list[str] = []
    lines.append("| " + " | ".join(cleaned[0]) + " |")
    lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
