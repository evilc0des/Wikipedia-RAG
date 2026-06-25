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


_PARALLEL_CHUNK_THRESHOLD = 200


def _stitch_and_write(children, sections, page_chunk, db, last_child_id,
                      last_section_id, last_page_id):
    if children:
        if last_child_id:
            children[0]["prev_id"] = last_child_id
            db.update_next_id(last_child_id, children[0]["chunk_id"])
        last_child_id = children[-1]["chunk_id"]
    if sections:
        if last_section_id:
            sections[0]["prev_id"] = last_section_id
            db.update_next_id(last_section_id, sections[0]["chunk_id"])
        last_section_id = sections[-1]["chunk_id"]
    if last_page_id:
        page_chunk["prev_id"] = last_page_id
        db.update_next_id(last_page_id, page_chunk["chunk_id"])
    last_page_id = page_chunk["chunk_id"]

    for c in children:
        db.insert_chunk(c)
    for sc in sections:
        db.insert_chunk(sc)
    db.insert_chunk(page_chunk)

    return last_child_id, last_section_id, last_page_id


def process_pages(pages, db, last_child_id=None, last_section_id=None,
                  last_page_id=None, page_count=0, workers=None,
                  commit_interval=100, progress=None):
    if not pages:
        return last_child_id, last_section_id, last_page_id, page_count

    import os
    w = workers if workers is not None else os.cpu_count()

    if len(pages) >= _PARALLEL_CHUNK_THRESHOLD and w > 1:
        from concurrent.futures import ProcessPoolExecutor

        chunksize = max(1, len(pages) // (w * 4))
        if progress:
            progress("parallel", len(pages), w, chunksize)

        with ProcessPoolExecutor(max_workers=w) as executor:
            for children, sections, page_chunk in executor.map(
                chunk_page, pages, chunksize=chunksize
            ):
                last_child_id, last_section_id, last_page_id = _stitch_and_write(
                    children, sections, page_chunk, db,
                    last_child_id, last_section_id, last_page_id,
                )
                page_count += 1
                if page_count % commit_interval == 0:
                    db.commit()
                    if progress:
                        progress("page", page_count, len(children), len(sections))
    else:
        if progress:
            progress("sequential", len(pages), 1, 1)

        for s in pages:
            children, sections, page_chunk = chunk_page(s)
            last_child_id, last_section_id, last_page_id = _stitch_and_write(
                children, sections, page_chunk, db,
                last_child_id, last_section_id, last_page_id,
            )
            page_count += 1
            if page_count % commit_interval == 0:
                db.commit()
                if progress:
                    progress("page", page_count, len(children), len(sections))

    return last_child_id, last_section_id, last_page_id, page_count


def chunk_page(page_data):
    s = page_data
    chunks = []
    sections = []
    last_chunk_len = 0

    for idx, para in enumerate(s["text"]["paragraph"]):
        section_title, section_path = parse_section(para)
        bullet = parse_bullet(para) if section_title is None else None

        if section_title is not None:
            if last_chunk_len == 0 and chunks:
                chunks.pop()
            chunk = create_new_chunk(
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
            )
            chunks.append(chunk)
            last_chunk_len = 0

        elif bullet is not None:
            if not chunks:
                chunk = create_new_chunk(
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
                )
                chunks.append(chunk)
            else:
                chunks[-1]["text"] += f"\n{bullet}"
                chunks[-1]["paragraph_end"] = idx
            last_chunk_len += 1

        else:
            text = para.strip()
            if not chunks:
                chunk = create_new_chunk(
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
                )
                chunks.append(chunk)
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
                section_chunk = create_new_chunk(
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
                )
                sections.append(section_chunk)
            else:
                sections[-1]["text"] += f"\n{last_chunk['text']}"
                sections[-1]["paragraph_end"] = idx
            last_chunk["parent_id"] = sections[-1]["chunk_id"]
            sections[-1]["children_ids"].append(last_chunk["chunk_id"])

    page_chunk = create_new_chunk(
        "page", "\n".join(s["text"] for s in sections),
        doc_id=s["wikipedia_id"],
        section_path=[s["wikipedia_title"]],
        title=s["wikipedia_title"],
        source_url=s["history"]["url"],
        paragraph_start=None,
        paragraph_end=None,
        prev_id=None,
        next_id=None,
        parent_chunk_id=None,
        children_ids=[sc["chunk_id"] for sc in sections],
    )
    return chunks, sections, page_chunk
