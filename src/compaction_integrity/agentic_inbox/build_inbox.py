"""Build the COMPINT-Inbox environment: AgentDojo's workspace inbox + real Enron mail.

AgentDojo's 31-email inbox is far too small to fill a context window, so we append real
messages from `Hellisotherpeople/enron_emails_parsed` (535,703 rows), whose
`date/from/to/subject/body` fields map onto AgentDojo's `Email` field for field.

Only `to` is modified, rewritten to the AgentDojo persona so the result is one coherent
mailbox and AgentDojo's contacts stay valid. Everything else is untouched real mail.

Usage::

    python -m compaction_integrity.agentic_inbox.build_inbox --n_enron 429 --seed 42 --out data/agentic_inbox/inbox_450.json
"""

import argparse
import datetime
import json
import random
from email.utils import parsedate_to_datetime
from pathlib import Path

import yaml
from agentdojo.default_suites.v1.tools.email_client import Inbox
from agentdojo.default_suites.v1.tools.types import Email, EmailStatus
from pydantic import EmailStr, TypeAdapter, ValidationError

ENRON_DATASET = "Hellisotherpeople/enron_emails_parsed"

# The same type AgentDojo's `Email` uses, so the filter and the model agree. Enron's
# maildir addresses include forms EmailStr rejects (`m..forney@enron.com`); those rows drop.
_ADDRESS_ADAPTER = TypeAdapter(EmailStr)

DEFAULT_OUT = Path("data/agentic_inbox/inbox.json")
# Streaming the corpus is the slow part, so the filtered pool is cached and reused. Stream
# order is deterministic, so an inbox built from the cache matches one built without it.
DEFAULT_POOL_CACHE = Path("data/agentic_inbox/enron_pool.jsonl")
DEFAULT_N_ENRON = 1000
DEFAULT_POOL = 20000
DEFAULT_SEED = 42
BODY_MIN_CHARS = 200
BODY_MAX_CHARS = 4000


def agentdojo_inbox_path() -> Path:
    import agentdojo

    return Path(agentdojo.__file__).parent / "data/suites/workspace/include/inbox.yaml"


def load_base_emails() -> tuple[str, list[dict]]:
    raw = yaml.safe_load(agentdojo_inbox_path().read_text())
    return raw["account_email"], raw["initial_emails"]


def clean_address(value: str) -> str | None:
    address = (value or "").strip().lower()
    try:
        return _ADDRESS_ADAPTER.validate_python(address)
    except ValidationError:
        return None


def clean_timestamp(value: str) -> datetime.datetime | None:
    """Some rows have body text spilled into `date`; those rows are dropped."""
    try:
        return parsedate_to_datetime(value).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def enron_rows(pool_size: int):
    """Stream the corpus and yield the first `pool_size` rows that pass the filters."""
    from datasets import load_dataset

    stream = load_dataset(ENRON_DATASET, split="train", streaming=True)
    seen: set[tuple[str, str]] = set()
    for row in stream:
        sender = clean_address(row["from"])
        recipient = clean_address(row["to"])
        timestamp = clean_timestamp(row["date"])
        subject = (row["subject"] or "").strip()
        body = (row["body"] or "").strip()
        if sender is None or recipient is None or timestamp is None or not subject:
            continue
        if not (BODY_MIN_CHARS <= len(body) <= BODY_MAX_CHARS):
            continue
        key = (sender, subject)
        if key in seen:
            continue
        seen.add(key)
        yield {"sender": sender, "subject": subject, "body": body, "timestamp": timestamp}
        if len(seen) >= pool_size:
            return


def to_email(row: dict, email_id: str, account_email: str) -> Email:
    return Email(
        id_=email_id,
        sender=row["sender"],
        recipients=[account_email],
        subject=row["subject"],
        body=row["body"],
        status=EmailStatus.received,
        read=False,
        timestamp=row["timestamp"],
    )


def load_pool(pool_size: int, cache_path: Path) -> list[dict]:
    """Filtered Enron rows, cached to JSONL and re-streamed only for a larger pool."""
    cached: list[dict] = []
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            row["timestamp"] = datetime.datetime.fromisoformat(row["timestamp"])
            cached.append(row)
    if len(cached) >= pool_size:
        return cached[:pool_size]

    print(f"streaming {ENRON_DATASET} for {pool_size} rows (cache has {len(cached)}) ...")
    pool = list(enron_rows(pool_size))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as handle:
        for row in pool:
            handle.write(json.dumps({**row, "timestamp": row["timestamp"].isoformat()}) + "\n")
    return pool


def build(n_enron: int, pool_size: int, seed: int, out_path: Path, cache_path: Path) -> Inbox:
    account_email, base_emails = load_base_emails()
    pool = load_pool(max(pool_size, n_enron), cache_path)
    sampled = random.Random(seed).sample(pool, n_enron)
    sampled.sort(key=lambda r: r["timestamp"])

    next_id = max(int(e["id_"]) for e in base_emails) + 1
    enron_emails = [
        to_email(row, str(next_id + offset), account_email) for offset, row in enumerate(sampled)
    ]

    inbox = Inbox(
        account_email=account_email,
        initial_emails=[Email.model_validate(e) for e in base_emails] + enron_emails,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(inbox.model_dump_json(indent=1))
    return inbox


def summarize(inbox: Inbox, n_base: int) -> str:
    received = inbox.received
    timestamps = [e.timestamp for e in inbox.emails.values()]
    body_chars = sum(len(e.body) for e in inbox.emails.values())
    return "\n".join(
        [
            f"account_email       {inbox.account_email}",
            f"emails              {len(inbox.emails)}  ({n_base} agentdojo + {len(inbox.emails) - n_base} enron)",
            f"received / sent     {len(received)} / {len(inbox.sent)}",
            f"contacts            {len(inbox.contact_list)}",
            f"timestamp range     {min(timestamps).date()} .. {max(timestamps).date()}",
            f"total body chars    {body_chars:,}  (~{body_chars // 4:,} tokens)",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_enron", type=int, default=DEFAULT_N_ENRON)
    parser.add_argument("--pool_size", type=int, default=DEFAULT_POOL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pool_cache", type=Path, default=DEFAULT_POOL_CACHE)
    args = parser.parse_args()

    _, base_emails = load_base_emails()
    inbox = build(args.n_enron, args.pool_size, args.seed, args.out, args.pool_cache)
    print(summarize(inbox, len(base_emails)))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
