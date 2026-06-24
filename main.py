import uuid

from datasets import load_dataset

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


ds = load_dataset("facebook/kilt_wikipedia", split="full", trust_remote_code=True, streaming=True)

pages = []
chunks = []
sections = []

for s in ds.take(2):
    last_chunk_len = 0

    for idx, para in enumerate(s["text"]["paragraph"]):
        section_title, section_path = parse_section(para)
        bullet = parse_bullet(para) if section_title is None else None

        if section_title is not None:
            if last_chunk_len == 0 and chunks:
                chunks.pop()
            chunks.append(create_new_chunk(
                "child", section_title,
                doc_id=s["wikipedia_id"],
                section_path=section_path,
                title=s["wikipedia_title"],
                source_url=s["history"]["url"],
                paragraph_start=idx,
                paragraph_end=idx,
                prev_id=_prev_id(chunks),
                next_id=None,
                parent_chunk_id=None,
            ))
            last_chunk_len = 0

        elif bullet is not None:
            if not chunks:
                chunks.append(create_new_chunk(
                    "child", bullet,
                    doc_id=s["wikipedia_id"],
                    section_path=[s["wikipedia_title"]],
                    title=s["wikipedia_title"],
                    source_url=s["history"]["url"],
                    paragraph_start=idx,
                    paragraph_end=idx,
                    prev_id=_prev_id(chunks),
                    next_id=None,
                    parent_id=None,
                ))
            else:
                chunks[-1]["text"] += f"\n{bullet}"
                chunks[-1]["paragraph_end"] = idx
            last_chunk_len += 1

        else:
            text = para.strip()
            if not chunks:
                chunks.append(create_new_chunk(
                    "child", text,
                    doc_id=s["wikipedia_id"],
                    section_path=[s["wikipedia_title"]],
                    title=s["wikipedia_title"],
                    source_url=s["history"]["url"],
                    paragraph_start=idx,
                    paragraph_end=idx,
                    prev_id=_prev_id(chunks),
                    next_id=None,
                    parent_id=None,
                ))
            else:
                chunks[-1]["text"] += f"\n{text}"
                chunks[-1]["paragraph_end"] = idx
            last_chunk_len += 1

        last_chunk = chunks[-1] if chunks else None
        last_section = sections[-1] if sections else None
        if last_chunk and last_chunk["section_path"] is not None:
            is_new = (
                last_section is None
                or last_chunk["section_path"][0] != last_section["section_path"][0]
            )
            if is_new:
                sections.append(create_new_chunk(
                    "section", last_chunk["text"],
                    doc_id=s["wikipedia_id"],
                    section_path=last_chunk["section_path"],
                    title=s["wikipedia_title"],
                    source_url=s["history"]["url"],
                    paragraph_start=idx,
                    paragraph_end=idx,
                    prev_id=_prev_id(sections),
                    next_id=None,
                    parent_id=None,
                ))
            else:
                sections[-1]["text"] += f"\n{last_chunk['text']}"
                sections[-1]["paragraph_end"] = idx
            last_chunk["parent_id"] = sections[-1]["chunk_id"]
            sections[-1]["children_ids"].append(last_chunk["chunk_id"])

    pages.append(create_new_chunk(
        "page", "\n".join(s["text"] for s in sections),
        doc_id=s["wikipedia_id"],
        section_path=[s["wikipedia_title"]],
        title=s["wikipedia_title"],
        source_url=s["history"]["url"],
        paragraph_start=None,
        paragraph_end=None,
        prev_id=_prev_id(pages),
        next_id=None,
        parent_chunk_id=None,
        children_ids=[s["chunk_id"] for s in sections],
    ))
    if len(pages) > 1:
        pages[-2]["next_id"] = pages[-1]["chunk_id"]

    print(f"Page {len(pages)}: {s['wikipedia_title']} — {len(sections)} sections, {len(chunks)} chunks")


# def build_indexes(chunks):
#     # 1. prepare texts + ids + metadata
#     # 2. build sparse index
#     # 3. build dense index
#     # 4. persist both
#     return sparse_index, dense_index, chunk_store