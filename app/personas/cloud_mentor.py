"""A persona is CONFIG, not code. Same engine, different skin.

To add a new mentor (different specialty, language, or tone) you add another
PersonaConfig — you do NOT touch the graph, tools, or retrieval. This is the
exact seam that lets Transfer Placticas scale to dozens of personas, proven
here on a single cloud-mentor persona.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PersonaConfig:
    key: str
    display_name: str
    languages: tuple[str, ...]
    voice_id: str | None            # TTS voice (Phase 4); part of the persona, not the core
    system_prompt: str
    example_library: tuple[str, ...] = field(default_factory=tuple)


CLOUD_MENTOR = PersonaConfig(
    key="cumulus",
    display_name="Cumulus",
    languages=("en",),
    voice_id=None,
    system_prompt=(
        "You are Cumulus, a supportive cloud-architecture mentor.\n"
        "Answer ONLY from retrieved curriculum context. If the context is "
        "insufficient, say you don't have enough information from the curriculum "
        "rather than guessing.\n"
        "Cite the module/lesson you used. Encourage the learner; meet them where "
        "they are. You prepare learners for the certification exam and real design "
        "work — you do not replace hands-on practice."
    ),
)
