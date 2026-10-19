from __future__ import annotations

from pathlib import Path

import pytest


def test_two_audio_template_slots_round_trip_independently(tmp_path: Path) -> None:
    try:
        from sf6_music_trigger.audio_templates import (
            AudioTemplate,
            AudioTemplateSide,
            load_audio_templates,
            save_audio_templates,
        )
    except ModuleNotFoundError:
        pytest.fail("audio template persistence is missing")

    template_1 = AudioTemplate(
        p1=AudioTemplateSide(True, "audio/p1-a.mp3", 0.23, "audio/p1-a-whiff.wav", 0.41),
        p2=AudioTemplateSide(False, "audio/p2-a.ogg", 0.34, "", 0.52),
    )
    template_2 = AudioTemplate(
        p1=AudioTemplateSide(False, "audio/p1-b.wav", 0.67, "", 0.18),
        p2=AudioTemplateSide(True, "audio/p2-b.mp3", 0.78, "audio/p2-b-whiff.ogg", 0.89),
    )
    target = tmp_path / "audio_templates.json"

    save_audio_templates(target, {"模板 1": template_1, "模板 2": template_2})

    assert load_audio_templates(target) == {
        "模板 1": template_1,
        "模板 2": template_2,
    }


def test_audio_template_names_can_be_added_and_renamed_without_losing_audio(
    tmp_path: Path,
) -> None:
    from sf6_music_trigger.audio_templates import (
        AudioTemplate,
        AudioTemplateSide,
        add_audio_template,
        load_audio_templates,
        rename_audio_template,
    )

    target = tmp_path / "audio_templates.json"
    original = AudioTemplate(
        p1=AudioTemplateSide(True, r"D:\Music\p1 theme.mp3", 0.23, "", 0.18),
        p2=AudioTemplateSide(False, "audio/p2.ogg", 0.34, "audio/miss.wav", 0.45),
    )

    add_audio_template(target, "  韩蛛莉  ", original)
    rename_audio_template(target, "韩蛛莉", "朱莉对战")

    assert load_audio_templates(target) == {"朱莉对战": original}


def test_audio_template_names_reject_blank_and_casefold_duplicates(tmp_path: Path) -> None:
    from sf6_music_trigger.audio_templates import (
        AudioTemplate,
        AudioTemplateSide,
        add_audio_template,
        rename_audio_template,
    )

    target = tmp_path / "audio_templates.json"
    template = AudioTemplate(AudioTemplateSide(), AudioTemplateSide())

    add_audio_template(target, "Ranked", template)
    with pytest.raises(ValueError, match="already exists"):
        add_audio_template(target, " ranked ", template)
    with pytest.raises(ValueError, match="nonblank"):
        add_audio_template(target, "   ", template)
    rename_audio_template(target, "Ranked", "RANKED")
    add_audio_template(target, "Casual", template)
    with pytest.raises(ValueError, match="already exists"):
        rename_audio_template(target, "RANKED", "casual")
