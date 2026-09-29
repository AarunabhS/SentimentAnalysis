"""Dataset verification must not replace local files or accept modified downloads."""
import io
import pytest
import fetch_data


def test_existing_different_input_is_left_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "ROOT", tmp_path)
    p = tmp_path/"data"/fetch_data.TARGET
    p.parent.mkdir()
    original = b"a local file that must be preserved"
    p.write_bytes(original)
    with pytest.raises(ValueError, match="left untouched"):
        fetch_data.fetch()
    assert p.read_bytes() == original


def test_unverified_download_is_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "ROOT", tmp_path)
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"tampered bytes"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_data.fetch()
    assert not (tmp_path/"data"/fetch_data.TARGET).exists()
