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

# Cosine-similarity floor below which retrieval is treated as "no real match",
# so the bot refuses instead of straining to answer off-curriculum questions.
# In-curriculum questions score ~0.65-0.82; unrelated ones top out near ~0.22.
RELEVANCE_FLOOR = float(os.environ.get("RELEVANCE_FLOOR", "0.35"))
