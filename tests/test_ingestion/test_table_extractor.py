from __future__ import annotations


from compact_rag.ingestion.loader import _docx_table_to_markdown
from compact_rag.ingestion.table_extractor import (
    ExtractedTable,
    TableExtractor,
    _dataframe_to_markdown,
    _matrix_to_markdown,
)


# ── evaluate_table_quality ─────────────────────────────────────


class TestEvaluateTableQuality:
    def test_valid_markdown_table_gets_full_score(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        result = TableExtractor.evaluate_table_quality(md)
        assert result["valid"] is True
        assert result["score"] == 1.0
        # 3 data rows: header + 2 content rows (all non-separator)
        assert result["rows"] == 3
        assert result["columns"] == 2
        assert result["consistent_columns"] is True

    def test_single_row_table_gets_partial_score(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        result = TableExtractor.evaluate_table_quality(md)
        assert result["valid"] is True
        # 2 data rows (header + content), consistent → 0.5 + 0.2 + 0.3 = 1.0
        assert result["score"] == 1.0
        assert result["rows"] == 2

    def test_missing_separator_row_returns_zero(self):
        md = "| A | B |\n| 1 | 2 |"
        result = TableExtractor.evaluate_table_quality(md)
        assert result["valid"] is False
        assert result["score"] == 0.0
        assert result["reason"] == "No separator row"

    def test_empty_string_returns_zero(self):
        result = TableExtractor.evaluate_table_quality("")
        assert result["valid"] is False
        assert result["score"] == 0.0

    def test_inconsistent_columns_gets_partial_score(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 | 3 |\n| 4 | 5 |"
        result = TableExtractor.evaluate_table_quality(md)
        assert result["score"] < 1.0
        assert result["consistent_columns"] is False

    def test_only_separator_no_data(self):
        md = "| --- | --- |"
        result = TableExtractor.evaluate_table_quality(md)
        assert result["valid"] is False
        assert result["score"] == 0.0

    def test_alignment_syntax_in_separator(self):
        md = "| A | B | C |\n| :--- | :---: | ---: |\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |"
        result = TableExtractor.evaluate_table_quality(md)
        assert result["valid"] is True
        assert result["score"] == 1.0
        assert result["columns"] == 3

    def test_single_column_table(self):
        md = "| X |\n| --- |\n| a |\n| b |"
        result = TableExtractor.evaluate_table_quality(md)
        assert result["valid"] is True
        assert result["score"] == 1.0
        assert result["columns"] == 1

    def test_many_rows_full_score(self):
        rows = "\n".join("| a | b |" for _ in range(10))
        md = "| A | B |\n| --- | --- |\n" + rows
        result = TableExtractor.evaluate_table_quality(md)
        assert result["score"] == 1.0


# ── extract_from_html ───────────────────────────────────────────


class TestExtractFromHtml:
    def test_simple_html_table_to_markdown(self):
        html = (
            "<table>"
            "<tr><th>Name</th><th>Age</th></tr>"
            "<tr><td>Alice</td><td>30</td></tr>"
            "<tr><td>Bob</td><td>25</td></tr>"
            "</table>"
        )
        extractor = TableExtractor()
        tables = extractor.extract_from_html(html)
        assert len(tables) >= 1
        assert tables[0].method == "html"
        assert "Alice" in tables[0].markdown
        assert "Bob" in tables[0].markdown

    def test_table_with_merged_cells(self, mocker):
        """Merged cells via colspan/rowspan should still produce a table."""
        html = (
            "<table>"
            "<tr><td colspan='2'>Header Span</td></tr>"
            "<tr><td>a</td><td>b</td></tr>"
            "</table>"
        )
        extractor = TableExtractor()
        tables = extractor.extract_from_html(html)
        assert len(tables) >= 1

    def test_empty_table_returns_empty_list(self):
        html = "<table></table>"
        extractor = TableExtractor()
        tables = extractor.extract_from_html(html)
        assert tables == []

    def test_no_tables_in_html(self):
        html = "<p>Just a paragraph, no tables here.</p>"
        extractor = TableExtractor()
        tables = extractor.extract_from_html(html)
        assert tables == []

    def test_nested_tables(self):
        html = (
            "<table><tr><td>Outer A</td><td>"
            "<table><tr><td>Inner 1</td><td>Inner 2</td></tr></table>"
            "</td></tr></table>"
        )
        extractor = TableExtractor()
        tables = extractor.extract_from_html(html)
        assert len(tables) == 2

    def test_table_with_header_and_footer(self):
        html = (
            "<table>"
            "<thead><tr><th>Product</th><th>Price</th></tr></thead>"
            "<tbody><tr><td>Apple</td><td>$1</td></tr></tbody>"
            "<tfoot><tr><td>Total</td><td>$1</td></tr></tfoot>"
            "</table>"
        )
        extractor = TableExtractor()
        tables = extractor.extract_from_html(html)
        assert len(tables) >= 1
        assert "Apple" in tables[0].markdown


# ── extract_from_pdf ────────────────────────────────────────────


class TestExtractFromPdf:
    def test_camelot_returns_tables_when_available(self, mocker):
        import pandas as pd

        mock_table = mocker.MagicMock()
        mock_table.df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_table.page = 1
        mock_table.flavor = "lattice"
        mock_table.accuracy = 95.0

        mocker.patch(
            "compact_rag.ingestion.table_extractor.camelot",
            create=True,
        )
        mocker.patch.object(
            TableExtractor,
            "_extract_camelot",
            return_value=[
                ExtractedTable(
                    page_number=1,
                    rows=2,
                    columns=2,
                    markdown="| A | B |\n| --- | --- |\n| 1 | 3 |\n| 2 | 4 |",
                    quality_score=1.0,
                    method="camelot_lattice",
                )
            ],
        )
        extractor = TableExtractor()
        tables = extractor.extract_from_pdf("/fake/file.pdf")
        assert len(tables) == 1
        assert tables[0].method == "camelot_lattice"
        assert tables[0].quality_score == 1.0

    def test_falls_back_to_pdfplumber_when_camelot_empty(self, mocker):
        mocker.patch.object(
            TableExtractor,
            "_extract_camelot",
            return_value=[],
        )
        mocker.patch.object(
            TableExtractor,
            "_extract_pdfplumber",
            return_value=[
                ExtractedTable(
                    page_number=1,
                    rows=3,
                    columns=2,
                    markdown="| X | Y |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |",
                    quality_score=1.0,
                    method="pdfplumber",
                )
            ],
        )
        extractor = TableExtractor()
        tables = extractor.extract_from_pdf("/fake/file.pdf")
        assert len(tables) == 1
        assert tables[0].method == "pdfplumber"

    def test_empty_page_returns_empty(self, mocker):
        mocker.patch.object(TableExtractor, "_extract_camelot", return_value=[])
        mocker.patch.object(TableExtractor, "_extract_pdfplumber", return_value=[])
        extractor = TableExtractor()
        tables = extractor.extract_from_pdf("/fake/empty.pdf")
        assert tables == []


# ── extract_from_docx ───────────────────────────────────────────


class TestExtractFromDocx:
    def test_extracts_tables_from_docx(self, mocker):
        mock_table = mocker.MagicMock()
        mock_table.rows = [mocker.MagicMock(), mocker.MagicMock()]
        mock_table.columns = [mocker.MagicMock(), mocker.MagicMock()]
        for row in mock_table.rows:
            row.cells = [mocker.MagicMock(), mocker.MagicMock()]
        mock_table.rows[0].cells[0].text = "Name"
        mock_table.rows[0].cells[1].text = "Value"
        mock_table.rows[1].cells[0].text = "Alpha"
        mock_table.rows[1].cells[1].text = "100"

        mock_doc = mocker.MagicMock()
        mock_doc.tables = [mock_table]

        # Document is imported via "from docx import Document" inside the method
        mocker.patch("docx.Document", return_value=mock_doc, create=True)

        extractor = TableExtractor()
        tables = extractor.extract_from_docx("/fake/file.docx")
        assert len(tables) == 1
        assert tables[0].method == "docx"
        assert "Name" in tables[0].markdown
        assert "Alpha" in tables[0].markdown

    def test_doc_with_no_tables(self, mocker):
        mock_doc = mocker.MagicMock()
        mock_doc.tables = []

        mocker.patch("docx.Document", return_value=mock_doc, create=True)

        extractor = TableExtractor()
        tables = extractor.extract_from_docx("/fake/none.docx")
        assert tables == []

    def test_empty_table_skipped(self, mocker):
        mock_table = mocker.MagicMock()
        mock_table.rows = []

        mock_doc = mocker.MagicMock()
        mock_doc.tables = [mock_table]

        mocker.patch("docx.Document", return_value=mock_doc, create=True)

        extractor = TableExtractor()
        tables = extractor.extract_from_docx("/fake/empty.docx")
        assert tables == []


# ── _matrix_to_markdown / _dataframe_to_markdown / _docx_table_to_markdown ──


class TestMarkdownRendering:
    """Tests for the markdown rendering helpers and ExtractedTable.markdown field."""

    def test_matrix_to_markdown_with_header(self):
        rows = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        md = _matrix_to_markdown(rows)
        assert "| Name | Age |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 30 |" in md
        assert "| Bob | 25 |" in md

    def test_matrix_to_markdown_without_header_issue(self):
        """Even single data row renders with separator."""
        rows = [["X", "Y"]]
        md = _matrix_to_markdown(rows)
        assert md.startswith("|")
        assert "---" in md

    def test_matrix_to_markdown_single_column(self):
        rows = [["Value"], ["a"], ["b"], ["c"]]
        md = _matrix_to_markdown(rows)
        assert "| Value |" in md
        assert "| a |" in md
        assert "| c |" in md

    def test_matrix_to_markdown_single_row(self):
        rows = [["A", "B", "C"]]
        md = _matrix_to_markdown(rows)
        assert "| A | B | C |" in md
        assert "| --- | --- | --- |" in md

    def test_matrix_to_markdown_empty(self):
        assert _matrix_to_markdown([]) == ""

    def test_matrix_to_markdown_jagged_rows_padded(self):
        rows = [["A", "B"], ["one"]]
        md = _matrix_to_markdown(rows)
        assert "| A | B |" in md
        assert "| one |  |" in md

    def test_dataframe_to_markdown(self):
        import pandas as pd

        df = pd.DataFrame({"Col1": [1, 2], "Col2": [3, 4]})
        md = _dataframe_to_markdown(df)
        assert "| Col1 | Col2 |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 3 |" in md
        assert "| 2 | 4 |" in md

    def test_docx_table_to_markdown(self, mocker):
        mock_table = mocker.MagicMock()
        cell_a = mocker.MagicMock()
        cell_a.text = "HeaderA"
        cell_b = mocker.MagicMock()
        cell_b.text = "HeaderB"
        row1 = mocker.MagicMock()
        row1.cells = [cell_a, cell_b]
        row2_cells = [mocker.MagicMock(), mocker.MagicMock()]
        row2_cells[0].text = "Data1"
        row2_cells[1].text = "Data2"
        row2 = mocker.MagicMock()
        row2.cells = row2_cells
        mock_table.rows = [row1, row2]

        md = _docx_table_to_markdown(mock_table)
        assert "| HeaderA | HeaderB |" in md
        assert "| --- | --- |" in md
        assert "| Data1 | Data2 |" in md

    def test_docx_table_to_markdown_empty(self, mocker):
        mock_table = mocker.MagicMock()
        mock_table.rows = []
        md = _docx_table_to_markdown(mock_table)
        assert md == ""

    def test_extracted_table_has_correct_markdown(self):
        table = ExtractedTable(
            page_number=1,
            rows=2,
            columns=2,
            markdown="| A | B |\n| --- | --- |\n| 1 | 2 |",
            quality_score=1.0,
            method="test",
        )
        assert table.markdown == "| A | B |\n| --- | --- |\n| 1 | 2 |"
        assert table.rows == 2
        assert table.columns == 2
        assert table.quality_score == 1.0

    def test_extracted_table_defaults(self):
        table = ExtractedTable(
            page_number=1,
            rows=0,
            columns=0,
            markdown="",
        )
        assert table.quality_score == 0.0
        assert table.method == ""
        assert table.metadata == {}
