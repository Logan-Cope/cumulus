"""Smoke tests. Real eval suite (hallucination/citation regression) comes later."""

def test_persona_config_loads():
    from app.personas.cloud_mentor import CLOUD_MENTOR
    assert CLOUD_MENTOR.display_name == "Cumulus"
    assert "en" in CLOUD_MENTOR.languages
