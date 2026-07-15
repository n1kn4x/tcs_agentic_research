"""PDF text extraction helpers for literature ingestion."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..artifact_store import ArtifactStore


class PDFTextExtractor:
    """Extract text from imported PDFs and store it next to the paper.

    The preferred backend is ``pypdf`` because it is pure Python. If it is unavailable, the
    extractor falls back to the common ``pdftotext`` command-line tool when installed.
    """

    def __init__(self, store: ArtifactStore):
        self.store = store

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

        text = self._extract_with_pypdf(source, page_separator=page_separator)
        if not text.strip():
            text = self._extract_with_pdftotext(source)
        if not text.strip():
            raise RuntimeError(
                f"No extractable text found in {source}. The PDF may be scanned; "
                "OCR is not implemented."
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
        try:
            from pypdf import PdfReader
        except Exception:
            return ""
        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_separator.format(page=idx) + page_text)
            return "".join(pages).strip() + ("\n" if pages else "")
        except Exception:
            return ""

    def _extract_with_pdftotext(self, pdf_path: Path) -> str:
        executable = _which("pdftotext")
        if executable is None:
            return ""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "paper.txt"
            result = subprocess.run(
                [executable, "-layout", str(pdf_path), str(output)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0 or not output.exists():
                return ""
            return output.read_text(encoding="utf-8", errors="replace")


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)
