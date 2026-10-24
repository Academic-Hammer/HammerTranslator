from __future__ import annotations

import unittest

import fitz

from hammer_translator.reflow.table import (
    HtmlCell,
    TableStructureError,
    _grid_rows_from_drawings,
    _pair_row_cells,
    _plain_table_text,
    _split_merged_row_labels,
)


class TableCellPairingTests(unittest.TestCase):
    def test_row_grid_is_recovered_from_per_row_vertical_rules(self):
        document = fitz.open()
        page = document.new_page(width=300, height=200)
        x_bounds = [50.0, 120.0, 200.0, 260.0]
        y_bounds = [40.0, 60.0, 80.0, 100.0]
        for y in (y_bounds[0], y_bounds[1], y_bounds[-1]):
            page.draw_line((x_bounds[0], y), (x_bounds[-1], y), width=0.5)
        for row in range(3):
            for x in x_bounds:
                page.draw_line((x, y_bounds[row] + 0.5), (x, y_bounds[row + 1] - 0.5), width=0.5)

        rows = _grid_rows_from_drawings(
            page,
            fitz.Rect(x_bounds[0], y_bounds[0], x_bounds[-1], y_bounds[-1]),
            3,
        )

        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(rows[0]), 3)
        self.assertEqual(rows[0][0], (50.0, 40.0, 120.0, 60.0))
        document.close()

    def test_overline_command_is_converted_instead_of_leaking(self):
        value = r"\overline { { | c _ { 0 , 1 } ^ { L _ { n } } - c _ { 0 , 1 } ^ { E } | } }"

        text = _plain_table_text(value)

        self.assertNotIn("overline", text)
        self.assertEqual(text, "| c₀,₁ᴸₙ - c₀,₁ᴱ |")

    def test_spaced_math_punctuation_is_normalized(self):
        value = r"<eq>7 . 3 4 ( . 0 4 5 , . 0 0 2 )</eq>"

        text = _plain_table_text(value)

        self.assertEqual(text, "7.34 (.045, .002)")

    def test_spaced_decimal_before_comma_is_normalized(self):
        value = r"<eq>3 . 6 1 \times 1 0 ^ { - 5 } ( 2 . , . 0 2 9 )</eq>"

        text = _plain_table_text(value)

        self.assertEqual(text, "3.61 × 10⁻⁵ (2., .029)")

    def test_empty_cells_stay_positional_when_counts_match(self):
        html_row = [
            HtmlCell(text="", colspan=1, rowspan=1),
            HtmlCell(text="5.51", colspan=1, rowspan=1),
            HtmlCell(text="9.69", colspan=1, rowspan=1),
        ]
        source_cells = [(20.0, 20.0, 40.0, 40.0), (40.0, 20.0, 80.0, 40.0), (80.0, 20.0, 120.0, 40.0)]

        paired = _pair_row_cells(html_row, source_cells)

        self.assertEqual(paired[0], ("", source_cells[0]))
        self.assertEqual(paired[1], ("5.51", source_cells[1]))
        self.assertEqual(paired[2], ("9.69", source_cells[2]))

    def test_consecutive_merged_row_labels_are_split(self):
        rows = [
            [HtmlCell("n", 1, 1), HtmlCell("value", 1, 1)],
            [HtmlCell("3", 1, 1), HtmlCell("a", 1, 1)],
            [HtmlCell("456", 1, 1), HtmlCell("b", 1, 1)],
            [HtmlCell("", 1, 1), HtmlCell("c", 1, 1)],
            [HtmlCell("", 1, 1), HtmlCell("d", 1, 1)],
            [HtmlCell("7", 1, 1), HtmlCell("e", 1, 1)],
        ]

        split = _split_merged_row_labels(rows)

        self.assertEqual([row[0].text for row in split[2:5]], ["4", "5", "6"])

    def test_leading_empty_html_cells_pair_with_merged_source_cell(self):
        html_row = [
            HtmlCell(text="", colspan=1, rowspan=1),
            HtmlCell(text="", colspan=1, rowspan=1),
            HtmlCell(text="LRP-γ 变体", colspan=1, rowspan=1),
        ]
        merged_source_cell = (300.0, 20.0, 400.0, 40.0)

        paired = _pair_row_cells(html_row, [merged_source_cell])

        self.assertEqual(paired, [("LRP-γ 变体", merged_source_cell)])

    def test_empty_html_cells_keep_later_columns_aligned(self):
        html_row = [
            HtmlCell(text="指标", colspan=1, rowspan=1),
            HtmlCell(text="", colspan=1, rowspan=1),
            HtmlCell(text="数值", colspan=1, rowspan=1),
        ]
        source_cells = [(20.0, 20.0, 120.0, 40.0), (220.0, 20.0, 300.0, 40.0)]

        paired = _pair_row_cells(html_row, source_cells)

        self.assertEqual(
            paired,
            [("指标", source_cells[0]), ("数值", source_cells[1])],
        )

    def test_ambiguous_nonempty_count_is_rejected_instead_of_shifting_columns(self):
        html_row = [
            HtmlCell(text="指标", colspan=1, rowspan=1),
            HtmlCell(text="训练", colspan=1, rowspan=1),
            HtmlCell(text="验证", colspan=1, rowspan=1),
        ]
        source_cells = [(20.0, 20.0, 80.0, 40.0), (80.0, 20.0, 140.0, 40.0)]

        with self.assertRaisesRegex(TableStructureError, "table_column_count_mismatch"):
            _pair_row_cells(html_row, source_cells)

    def test_colspan_uses_the_merged_destination_box(self):
        html_row = [
            HtmlCell(text="合并标题", colspan=2, rowspan=1),
            HtmlCell(text="数值", colspan=1, rowspan=1),
        ]
        source_cells = [
            (20.0, 20.0, 80.0, 40.0),
            (80.0, 20.0, 140.0, 40.0),
            (140.0, 20.0, 200.0, 40.0),
        ]

        paired = _pair_row_cells(html_row, source_cells)

        self.assertEqual(paired[0][1], (20.0, 20.0, 140.0, 40.0))
        self.assertEqual(paired[1], ("数值", source_cells[2]))

    def test_span_grid_mismatch_is_rejected(self):
        html_row = [
            HtmlCell(text="合并标题", colspan=2, rowspan=1),
            HtmlCell(text="数值", colspan=1, rowspan=1),
        ]
        source_cells = [(20.0, 20.0, 80.0, 40.0), (80.0, 20.0, 140.0, 40.0)]

        with self.assertRaisesRegex(TableStructureError, "table_span_grid_mismatch"):
            _pair_row_cells(html_row, source_cells)


if __name__ == "__main__":
    unittest.main()
