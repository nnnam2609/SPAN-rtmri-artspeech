from __future__ import annotations

from speech2rtmri_artspeech.paths import ConfiguredPathMapper


def test_server_to_local_mapping() -> None:
    mapper = ConfiguredPathMapper([
        {
            "source": "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi",
            "target": "C:/Users/nhnguyen/PhD_A2A/iadi",
        }
    ])
    mapped = mapper.map_path_string(
        "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2/1775/S10"
    )
    assert mapped == "C:/Users/nhnguyen/PhD_A2A/iadi/ArtSpeech_Database_2/1775/S10"
