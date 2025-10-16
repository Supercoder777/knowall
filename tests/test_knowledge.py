from pathlib import Path

from tools.knowledge import chunk_text, read_document


def test_chunk_text_splits_long_passage():
    text = "Lorem ipsum " * 200
    chunks = chunk_text(text, max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_read_document_txt(tmp_path: Path):
    doc = tmp_path / "note.txt"
    doc.write_text("sample content")
    content = read_document(doc)
    assert "sample" in content
