"""Real-time controller output recorder.

The recorder writes each sample to a temporary CSV file immediately, then
converts that CSV into a normal .xlsx workbook when recording stops.  It uses
only the Python standard library so the ROS package does not need an extra
Excel dependency on the vehicle computer.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import tempfile
import zipfile
from xml.sax.saxutils import escape


class OutputExcelRecorder:
    """Record controller output samples and save them as an Excel workbook."""

    OUTPUT_HEADERS = [
        "ros_time_sec",
        "record_elapsed_sec",
        "loop_count",
        "local_x_m",
        "local_y_m",
        "local_heading_rad",
        "yaw_rate_rad_s",
        "speed_x_m_s",
        "last_yaw_rate_rad_s",
        "last_speed_x_m_s",
        "last_left_wheel_cmd_rad_s",
        "last_right_wheel_cmd_rad_s",
        "left_wheel_feedback_rad_s",
        "right_wheel_feedback_rad_s",
        "left_wheel_feedback_rpm",
        "right_wheel_feedback_rpm",
        "out_00_f_U_estimated_g_U",
        "out_01_f_R_estimated_g_R",
        "out_02_f_U_observed_Obserdata_U",
        "out_03_f_R_observed_Obserdata_R",
        "out_04_runtime_A00",
        "out_05_runtime_A11",
        "out_06_runtime_B00",
        "out_07_runtime_B01",
        "out_08_runtime_B10",
        "out_09_runtime_B11",
        "out_10_runtime_A01",
        "out_11_runtime_A10",
        "out_12_left_wheel_cmd_rad_s",
        "out_13_right_wheel_cmd_rad_s",
        "out_14_ref_X_m",
        "out_15_ref_Y_m",
        "out_16_ref_Psi_rad",
        "out_17_lateral_error_m",
        "out_18_checkU",
        "out_19_checkR",
    ]

    def __init__(self, record_dir="~/lv_tan_cyh/record", logger=None):
        self.record_dir = Path(record_dir).expanduser()
        self.logger = logger
        self.is_recording = False
        self.start_time_sec = None
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self.xlsx_path = None
        self.row_count = 0

    def start(self, ros_time_sec=None):
        """Start a new recording file. Existing Excel files are never removed."""

        if self.is_recording:
            return self.xlsx_path

        self.record_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"output_record_{timestamp}"
        self.xlsx_path = self._unique_path(self.record_dir / f"{base_name}.xlsx")
        self.csv_path = self.xlsx_path.with_suffix(".tmp.csv")

        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self.OUTPUT_HEADERS)
        self.csv_file.flush()

        self.start_time_sec = self._to_float_or_none(ros_time_sec)
        self.row_count = 0
        self.is_recording = True

        self._log_info(f"Started output recording: {self.xlsx_path}")
        return self.xlsx_path

    def append_output_sample(
        self,
        ros_time_sec,
        loop_count,
        local_x,
        local_y,
        local_heading,
        yaw_rate,
        speed_x,
        last_yaw_rate,
        last_speed_x,
        last_left_wheel_cmd,
        last_right_wheel_cmd,
        output_values,
        left_wheel_feedback=math.nan,
        right_wheel_feedback=math.nan,
        left_wheel_feedback_rpm=math.nan,
        right_wheel_feedback_rpm=math.nan,
    ):
        """Append one controller.output(...) sample and flush it to disk."""

        if not self.is_recording or self.csv_writer is None:
            return

        ros_time_sec = self._to_float_or_none(ros_time_sec)
        if self.start_time_sec is None and ros_time_sec is not None:
            self.start_time_sec = ros_time_sec

        elapsed = ""
        if ros_time_sec is not None and self.start_time_sec is not None:
            elapsed = ros_time_sec - self.start_time_sec

        values = list(output_values) if output_values is not None else []
        if len(values) < 20:
            values.extend([math.nan] * (20 - len(values)))
        else:
            values = values[:20]

        row = [
            ros_time_sec,
            elapsed,
            loop_count,
            local_x,
            local_y,
            local_heading,
            yaw_rate,
            speed_x,
            last_yaw_rate,
            last_speed_x,
            last_left_wheel_cmd,
            last_right_wheel_cmd,
            left_wheel_feedback,
            right_wheel_feedback,
            left_wheel_feedback_rpm,
            right_wheel_feedback_rpm,
            *values,
        ]

        self.csv_writer.writerow([self._csv_value(value) for value in row])
        self.csv_file.flush()
        self.row_count += 1

    def stop(self):
        """Stop recording and convert the temporary CSV into an .xlsx file."""

        if not self.is_recording:
            return None

        self.is_recording = False
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        if self.csv_path is not None and self.xlsx_path is not None:
            self._csv_to_xlsx(self.csv_path, self.xlsx_path)
            try:
                self.csv_path.unlink()
            except OSError:
                pass
            self._log_info(
                "Saved output recording: "
                f"{self.xlsx_path} ({self.row_count} samples)"
            )
            return self.xlsx_path

        return None

    close = stop

    def _unique_path(self, path):
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        for index in range(1, 1000):
            candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not create unique record filename for {path}")

    @staticmethod
    def _to_float_or_none(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _csv_value(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "" if value is None else value
        return value if math.isfinite(value) else ""

    def _log_info(self, message):
        if self.logger is not None:
            try:
                self.logger.info(message)
                return
            except Exception:
                pass
        print(message)

    @staticmethod
    def _column_name(index):
        name = ""
        index += 1
        while index:
            index, remainder = divmod(index - 1, 26)
            name = chr(65 + remainder) + name
        return name

    @staticmethod
    def _looks_numeric(text):
        if text == "":
            return False
        try:
            value = float(text)
        except ValueError:
            return False
        return math.isfinite(value)

    @classmethod
    def _cell_xml(cls, row_index, col_index, value, style_id=None):
        ref = f"{cls._column_name(col_index)}{row_index}"
        style = f' s="{style_id}"' if style_id is not None else ""

        if value == "":
            return f'<c r="{ref}"{style}/>'

        if cls._looks_numeric(value):
            return f'<c r="{ref}"{style}><v>{value}</v></c>'

        safe_text = escape(str(value))
        return (
            f'<c r="{ref}" t="inlineStr"{style}>'
            f"<is><t>{safe_text}</t></is></c>"
        )

    @classmethod
    def _csv_to_xlsx(cls, csv_path, xlsx_path):
        max_rows = 0
        max_cols = 1
        with csv_path.open("r", newline="", encoding="utf-8") as source:
            for row in csv.reader(source):
                max_rows += 1
                max_cols = max(max_cols, len(row))

        max_rows = max(max_rows, 1)
        last_col = cls._column_name(max_cols - 1)
        dimension = f"A1:{last_col}{max_rows}"
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        with zipfile.ZipFile(
            xlsx_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                "[Content_Types].xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
                "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
                "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
                "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
                "<Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
                "<Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>"
                "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
                "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>"
                "</Types>",
            )
            archive.writestr(
                "_rels/.rels",
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
                "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>"
                "<Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/>"
                "</Relationships>",
            )
            archive.writestr(
                "docProps/core.xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
                "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
                "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
                "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
                "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
                "<dc:creator>learning_preview_controller</dc:creator>"
                f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{created}</dcterms:created>"
                f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{created}</dcterms:modified>"
                "</cp:coreProperties>",
            )
            archive.writestr(
                "docProps/app.xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" "
                "xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">"
                "<Application>learning_preview_controller</Application>"
                "</Properties>",
            )
            archive.writestr(
                "xl/workbook.xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
                "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
                "<sheets><sheet name=\"OutputRecord\" sheetId=\"1\" r:id=\"rId1\"/></sheets>"
                "</workbook>",
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>"
                "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>"
                "</Relationships>",
            )
            archive.writestr(
                "xl/styles.xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
                "<fonts count=\"2\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font>"
                "<font><b/><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
                "<fills count=\"2\"><fill><patternFill patternType=\"none\"/></fill>"
                "<fill><patternFill patternType=\"gray125\"/></fill></fills>"
                "<borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"
                "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
                "<cellXfs count=\"2\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/>"
                "<xf numFmtId=\"0\" fontId=\"1\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyFont=\"1\"/></cellXfs>"
                "</styleSheet>",
            )

            with archive.open("xl/worksheets/sheet1.xml", "w") as sheet:
                def write(text):
                    sheet.write(text.encode("utf-8"))

                write(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                    "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
                    "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
                    f"<dimension ref=\"{dimension}\"/>"
                    "<sheetViews><sheetView workbookViewId=\"0\">"
                    "<pane ySplit=\"1\" topLeftCell=\"A2\" activePane=\"bottomLeft\" state=\"frozen\"/>"
                    "<selection pane=\"bottomLeft\"/>"
                    "</sheetView></sheetViews>"
                    "<sheetFormatPr defaultRowHeight=\"15\"/>"
                    "<cols>"
                )

                for col_index in range(max_cols):
                    width = 18
                    if col_index == 0:
                        width = 16
                    elif col_index == 1:
                        width = 18
                    elif col_index >= 16:
                        width = 24
                    write(
                        f'<col min="{col_index + 1}" max="{col_index + 1}" '
                        f'width="{width}" customWidth="1"/>'
                    )

                write("</cols><sheetData>")

                with csv_path.open("r", newline="", encoding="utf-8") as source:
                    for row_index, row in enumerate(csv.reader(source), start=1):
                        if len(row) < max_cols:
                            row = row + [""] * (max_cols - len(row))
                        write(f'<row r="{row_index}">')
                        for col_index, value in enumerate(row):
                            style_id = 1 if row_index == 1 else None
                            write(cls._cell_xml(row_index, col_index, value, style_id))
                        write("</row>")

                write(
                    "</sheetData>"
                    f"<autoFilter ref=\"{dimension}\"/>"
                    "</worksheet>"
                )
