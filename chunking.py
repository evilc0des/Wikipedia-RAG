import uuid

SECTION_PREFIX = "Section::::"
BULLET_PREFIX = "BULLET::::"


def parse_section(text):
    if text.startswith(SECTION_PREFIX):
        title = text[len(SECTION_PREFIX):].strip()
        return title, title.split(":")
    return None, None


def parse_bullet(text):
    if text.startswith(BULLET_PREFIX):
        return f"- {text[len(BULLET_PREFIX):].strip()}"
    return None


def create_new_chunk(chunk_type, text, **metadata):
    return {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": metadata.get("doc_id"),
        "chunk_type": chunk_type,
        "text": text,
        "section_path": metadata.get("section_path"),
        "title": metadata.get("title"),
        "source_url": metadata.get("source_url"),
        "paragraph_start": metadata.get("paragraph_start"),
        "paragraph_end": metadata.get("paragraph_end"),
        "prev_id": metadata.get("prev_id"),
        "next_id": metadata.get("next_id"),
        "parent_id": metadata.get("parent_chunk_id"),
        "children_ids": [],
    }


def _prev_id(items):
    return items[-1]["chunk_id"] if items else None
