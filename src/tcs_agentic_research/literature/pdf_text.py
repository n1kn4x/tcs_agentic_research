"""PDF text extraction helpers for literature ingestion."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ..artifact_store import ArtifactStore


class PDFTextExtractor:
    """Extract text from imported PDFs and store it next to the paper.

    The preferred backend is Poppler ``pdftotext -layout`` because theorem labels and paragraph
    boundaries matter more than dependency purity here. The extractor falls back to ``pypdf`` and
    then optional OCR backends: ``ocrmypdf`` or ``pdftoppm`` plus ``tesseract``.
    """

    def __init__(self, store: ArtifactStore):
        self.store = store
        self._last_pdf_warnings: list[str] = []

    def extract_pdf_text(
        self,
        pdf_path: str | Path,
        *,
        output_path: str | Path | None = None,
        page_separator: str = "\n\n--- page {page} ---\n\n",
    ) -> str:
        """Extract text from ``pdf_path`` and write a UTF-8 text artifact.

        ``pdf_path`` may be a workspace-relative artifact path or an absolute local path. The
        output is always written inside the workspace; if omitted, it is placed beside the PDF
        for workspace PDFs or under ``LiteratureDB/papers/extracted_text/`` for external PDFs.
        """
        source = Path(pdf_path).expanduser()
        workspace_source = False
        if not source.is_absolute():
            source = self.store.resolve(source)
            workspace_source = True
        else:
            source = source.resolve()
            try:
                source.relative_to(self.store.root)
                workspace_source = True
            except ValueError:
                workspace_source = False
        if not source.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text = self._extract_with_pdftotext(source, page_separator=page_separator)
        if not text.strip():
            text = self._extract_with_pypdf(source, page_separator=page_separator)
            if self._last_pdf_warnings:
                self.store.append_jsonl(
                    "LiteratureDB/pdf_warnings.jsonl",
                    {
                        "pdf_path": self.store.relpath(source) if workspace_source else str(source),
                        "warning_count": len(self._last_pdf_warnings),
                        "warnings": self._last_pdf_warnings[:20],
                        "summary": "Non-fatal PDF parser warnings emitted by pypdf and suppressed from stderr.",
                    },
                )
        if not text.strip():
            text = self._extract_with_ocr(source, page_separator=page_separator)
        if not text.strip():
            raise RuntimeError(
                f"No extractable text found in {source}. Tried pypdf, pdftotext, and OCR "
                "backends. Install `ocrmypdf` or `pdftoppm` plus `tesseract` for scanned PDFs."
            )

        if output_path is None:
            if workspace_source:
                rel_source = self.store.relpath(source)
                output_path = str(Path(rel_source).with_suffix(".txt"))
            else:
                output_path = f"LiteratureDB/papers/extracted_text/{source.stem}.txt"
        self.store.write_text(output_path, text)
        return self.store.relpath(self.store.resolve(output_path))

    def _extract_with_pypdf(self, pdf_path: Path, *, page_separator: str) -> str:
        self._last_pdf_warnings = []
        try:
            from pypdf import PdfReader
        except Exception:
            return ""
        try:
            with _capture_pypdf_warnings() as warnings:
                reader = PdfReader(str(pdf_path))
                pages = []
                for idx, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append(page_separator.format(page=idx) + page_text)
                self._last_pdf_warnings = list(dict.fromkeys(warnings))
            return "".join(pages).strip() + ("\n" if pages else "")
        except Exception:
            return ""

    def _extract_with_pdftotext(
        self, pdf_path: Path, *, page_separator: str = "\n\n--- page {page} ---\n\n"
    ) -> str:
        executable = _which("pdftotext")
        if executable is None:
            return ""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "paper.txt"
            try:
                result = subprocess.run(
                    [executable, "-layout", str(pdf_path), str(output)],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300,
                )
            except (OSError, subprocess.TimeoutExpired):
                return ""
            if result.returncode != 0 or not output.exists():
                return ""
            raw = output.read_text(encoding="utf-8", errors="replace")
            pages = raw.split("\f")
            rendered = [
                page_separator.format(page=index) + page.strip()
                for index, page in enumerate(pages, 1)
                if page.strip()
            ]
            return "".join(rendered).strip() + ("\n" if rendered else "")

    def _extract_with_ocr(self, pdf_path: Path, *, page_separator: str) -> str:
        """Best-effort OCR extraction for scanned PDFs.

        This method intentionally depends only on command-line tools, so OCR support is optional
        and does not make the base Python package heavier. It first tries ``ocrmypdf`` because
        it keeps page layout well, then renders pages with ``pdftoppm`` and runs ``tesseract``
        on each page image.
        """
        text = self._extract_with_ocrmypdf(pdf_path, page_separator=page_separator)
        if text.strip():
            return text
        return self._extract_with_tesseract(pdf_path, page_separator=page_separator)

    def _extract_with_ocrmypdf(self, pdf_path: Path, *, page_separator: str) -> str:
        executable = _which("ocrmypdf")
        if executable is None:
            return ""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_pdf = Path(tmpdir) / "ocr.pdf"
            try:
                result = subprocess.run(
                    [
                        executable,
                        "--quiet",
                        "--skip-text",
                        str(pdf_path),
                        str(output_pdf),
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=900,
                )
            except (OSError, subprocess.TimeoutExpired):
                return ""
            if result.returncode != 0 or not output_pdf.exists():
                return ""
            text = self._extract_with_pdftotext(output_pdf)
            if text.strip():
                return text
            return self._extract_with_pypdf(output_pdf, page_separator=page_separator)

    def _extract_with_tesseract(self, pdf_path: Path, *, page_separator: str) -> str:
        pdftoppm = _which("pdftoppm")
        tesseract = _which("tesseract")
        if pdftoppm is None or tesseract is None:
            return ""
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "page"
            try:
                render = subprocess.run(
                    [pdftoppm, "-png", "-r", "200", str(pdf_path), str(prefix)],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=900,
                )
            except (OSError, subprocess.TimeoutExpired):
                return ""
            if render.returncode != 0:
                return ""
            images = sorted(Path(tmpdir).glob("page-*.png"), key=_page_image_sort_key)
            if not images:
                return ""

            pages: list[str] = []
            for idx, image in enumerate(images, start=1):
                try:
                    ocr = subprocess.run(
                        [tesseract, str(image), "stdout", "--psm", "1"],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=180,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if ocr.returncode != 0:
                    continue
                page_text = _clean_ocr_text(ocr.stdout)
                if page_text:
                    pages.append(page_separator.format(page=idx) + page_text)
            return "".join(pages).strip() + ("\n" if pages else "")


@contextmanager
def _capture_pypdf_warnings():
    """Capture noisy non-fatal pypdf parser warnings without printing them to stderr."""
    captured: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - logging callback
            if record.levelno >= logging.WARNING:
                captured.append(record.getMessage())

    loggers = [logging.getLogger("pypdf"), logging.getLogger("pypdf._reader")]
    handler = _Handler(level=logging.WARNING)
    previous = [(logger, logger.level, logger.propagate) for logger in loggers]
    try:
        for logger in loggers:
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
        yield captured
    finally:
        for logger, level, propagate in previous:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate


def _page_image_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    suffix = stem.rsplit("-", 1)[-1]
    page_number = int(suffix) if suffix.isdigit() else 0
    return page_number, path.name


def _clean_ocr_text(text: str) -> str:
    return text.replace("\f", "").strip()


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)
