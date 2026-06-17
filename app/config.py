"""Central config. Everything is env-driven so models/keys are swappable."""
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/cumulus")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-6")
REASONING_MODEL = os.environ.get("REASONING_MODEL", "claude-opus-4-8")
CHEAP_MODEL = os.environ.get("CHEAP_MODEL", "claude-haiku-4-5-20251001")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "6"))

# Two cosine-similarity floors, deliberately separate:
#   RELEVANCE_FLOOR - below this the TOP hit is "no real match", so the bot
#     refuses rather than strain to answer an off-curriculum question. Kept low
#     so paraphrased-but-valid questions still get answered.
#   CONTEXT_FLOOR - only chunks at/above this bar are put into the prompt and
#     cited. Higher than the refusal floor so weak, tangential chunks stop
#     leaking into citations (raises citation precision). The top hit is always
#     kept as a fallback so a question we chose to answer never has empty context.
RELEVANCE_FLOOR = float(os.environ.get("RELEVANCE_FLOOR", "0.35"))
CONTEXT_FLOOR = float(os.environ.get("CONTEXT_FLOOR", "0.55"))
