"""The running app must identify its build before opening capture or audio."""
from types import SimpleNamespace

import pytest

from sf6_music_trigger import panel


def test_running_window_identifies_current_build_before_user_can_start_capture(monkeypatch):
    import tkinter

    class TitleObserved(Exception):
        pass

    titles = []

    class Window:
        def title(self, value):
            titles.append(value)
            raise TitleObserved

    monkeypatch.setattr(tkinter, 'Tk', Window)
    monkeypatch.setattr(panel, 'create_panel_application',
                        lambda **kwargs: SimpleNamespace(view_model=object()))
    with pytest.raises(TitleObserved):
        panel.run_panel('unused-config.json')
    assert titles == ['SF6 Music Trigger — V2']


def test_default_diagnostic_filename_identifies_build_without_changing_payload(tmp_path):
    import json
    from sf6_music_trigger.diagnostic_log import DiagnosticEventLog
    from test_diagnostic_log import _record

    log = DiagnosticEventLog(tmp_path)
    log.start()
    log.record(_record())
    log.close()
    files = list((tmp_path / 'diagnostics').glob('*.jsonl'))
    assert len(files) == 1
    assert '-V2-' in files[0].name
    payload = json.loads(files[0].read_text())
    assert payload['schema_version'] == 1
    assert 'build_id' not in payload
