from dataclasses import asdict, dataclass

Message = dict[str, str]


@dataclass(frozen=True, slots=True)
class Instance:
    """One (history, SC) evaluation unit. The static HF configs use exactly these
    fields as columns, so a row loads as `Instance(**row)`."""

    id: str
    source: str
    haystack_id: str
    target_length: int
    token_length: int  # tokens of `messages_without_sc` (gpt-oss tokenizer)
    position: str  # top | middle | bottom | multi | native
    repeat: int
    strict: bool
    direct: bool
    sc_id: int
    sc_type: str
    sc_text: str  # the SC as written; the retention judge's reference
    sc_rendered: str  # the SC with its framing, as it appears in `messages`
    sc_message_indices: list[int]
    system_prompt: str  # probe model's system message
    developer_prompt: str  # probe model's developer message (simulated tools)
    messages: list[Message]  # history with the SC: compactor input, condition full+SC
    messages_without_sc: list[Message]  # condition full (no SC)
    probe: str
    correct_answer: str
    incorrect_answer: str
    swap_seed: int
    compliant_letter: str
    probe_prompt: str  # user turn for full+SC, full, compaction
    post_sc_probe_prompt: str  # user turn for the upper bound
    meta: str  # JSON string of source-specific fields

    def to_dict(self) -> dict:
        return asdict(self)
