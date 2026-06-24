import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path, override=True)


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


class AnswerGenerator:
    def __init__(self, config=None):
        config = config or {}
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.api_base = config.get("api_base") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.extra_headers = config.get("headers") or {}

    def generate(self, query_text, context_blocks):
        if not context_blocks:
            return {
                "answer_text": None,
                "citations": [],
                "grounded": True,
                "abstained": True,
                "reason": "No context blocks provided",
            }

        context_str = self._format_context(context_blocks)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context blocks:\n\n{context_str}\n\nQuestion: {query_text}"},
        ]

        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        print(headers)

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = requests.post(url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        raw = data["choices"][0]["message"]["content"] or ""

        abstained = raw.strip().upper().startswith("ABSTAIN:")
        conflict = raw.strip().upper().startswith("CONFLICT:")

        if abstained:
            return {
                "answer_text": None,
                "citations": [],
                "grounded": True,
                "abstained": True,
                "reason": raw.strip()[len("ABSTAIN:"):].strip() or None,
            }

        if conflict:
            return {
                "answer_text": raw.strip(),
                "citations": [],
                "grounded": False,
                "abstained": False,
                "reason": raw.strip(),
            }

        citations = self._extract_citations(raw, context_blocks)
        grounded, reason = self._validate_citations(raw, citations, context_blocks)

        return {
            "answer_text": raw.strip(),
            "citations": citations,
            "grounded": grounded,
            "abstained": False,
            "reason": reason,
        }

    def _format_context(self, context_blocks):
        parts = []
        for i, block in enumerate(context_blocks):
            block_id = f"[S{i:02d}]"
            parts.append(f"{block_id} {block['text']}")
        return "\n\n".join(parts)

    def _extract_citations(self, answer_text, context_blocks):
        seen = set()
        citations = []
        for match in _CITATION_PATTERN.finditer(answer_text):
            idx = int(match.group(1))
            cid = f"S{idx:02d}"
            if cid in seen:
                continue
            seen.add(cid)
            if idx < len(context_blocks):
                block = context_blocks[idx]
                citations.append({
                    "citation_id": cid,
                    "source_id": block.get("source_id"),
                    "section_id": block.get("section_id"),
                    "supporting_child_ids": block.get("supporting_child_ids", []),
                })
        return citations

    def _validate_citations(self, answer_text, citations, context_blocks):
        reasons = []

        cited_indices = {int(m.group(1)) for m in _CITATION_PATTERN.finditer(answer_text)}
        out_of_bounds = [i for i in cited_indices if i >= len(context_blocks)]
        if out_of_bounds:
            reasons.append(f"Citation indices out of bounds: {out_of_bounds}")

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
        factual_sentences = [
            s for s in sentences
            if not s.upper().startswith(("ABSTAIN:", "CONFLICT:"))
        ]
        uncited = [
            s[:80] for s in factual_sentences
            if not _CITATION_PATTERN.search(s)
        ]
        if uncited and factual_sentences:
            reasons.append(f"Sentences without citations: {len(uncited)}")

        if not citations and not any(
            answer_text.strip().upper().startswith(p) for p in ("ABSTAIN:", "CONFLICT:")
        ):
            reasons.append("No citations found in generated answer")

        grounded = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else None
        return grounded, reason


def build_context_blocks(sections):
    blocks = []
    for section in sections:
        blocks.append({
            "source_id": section.get("doc_id"),
            "section_id": section.get("chunk_id"),
            "section_path": section.get("section_path"),
            "text": section.get("text", ""),
            "supporting_child_ids": section.get("child_ids", []),
            "retrieval_score": section.get("retrieval_score", 0.0),
            "rerank_score": section.get("rerank_score", section.get("score", 0.0)),
        })
    return blocks


_SYSTEM_PROMPT = """\
You are a precise answer generator. Follow these rules strictly:

1. Answer the question using ONLY facts from the provided context blocks.
2. Each context block is prefixed with its index like [S00], [S01], etc.
3. Every factual sentence in your answer MUST end with the citation marker of the \
context block(s) it uses. Example: "The letter A is a vowel. [S00]"
4. If multiple blocks support the same claim, list all: [S00][S02]
5. If the context blocks do NOT contain sufficient evidence to answer, respond with \
exactly: ABSTAIN:
6. If the context blocks contain conflicting information, state the conflict clearly \
and respond with: CONFLICT:
7. Do not use any knowledge outside the provided context blocks.
8. Do not output a citation unless the cited block directly supports the claim.
9. Keep answers concise and factual."""
