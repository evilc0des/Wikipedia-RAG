import uuid

from datasets import load_dataset

ds = load_dataset("facebook/kilt_wikipedia", split="full", trust_remote_code=True, streaming=True)

sample = list(ds.take(1))

def is_new_section(text: str):
    if text.startswith("Section::::"):
        return True
    return False

def get_section_title(text: str):
    if is_new_section(text):
        return text.split("Section::::")[1].strip()
    return None

def is_bullet_point(text: str):
    if text.startswith("BULLET::::"):
        return True
    return False

def get_bullet_point(text: str):
    if is_bullet_point(text):
        return f"- {text.split('BULLET::::')[1].strip()}"
    return None

def get_section_path(text: str):
    if is_new_section(text):
        return text.split("Section::::")[1].strip().split(":")
    return None

def create_new_chunk(type, text, metadata):
    return {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": metadata.get("doc_id", None),
        "chunk_type": type,
        "text": text,
        "section_path": metadata.get("section_path", None),
        "title": metadata.get("title", None),
        "source_url": metadata.get("source_url", None),
        "paragraph_start": metadata.get("paragraph_start", None),
        "paragraph_end": metadata.get("paragraph_end", None),
        "prev_id": metadata.get("previous_chunk_id", None),
        "next_id": metadata.get("next_chunk_id", None),
        "parent_id": metadata.get("parent_chunk_id", None),
        "children_ids": [],
    }

chunks = []
sections = []
pages = []

for s in sample:
    last_chunk_len = 0
    current_section = None
    for idx, para in enumerate(s["text"]["paragraph"]):
        if(is_new_section(para)):
            if(last_chunk_len == 0):
                chunks.pop();
            section_path = get_section_path(para)
            chunks.append(create_new_chunk("child", get_section_title(para), {
                "doc_id": s["wikipedia_id"],
                "section_path": section_path,
                "title": s["wikipedia_title"],
                "source_url": s["history"]["url"],
                "paragraph_start": idx,
                "paragraph_end": idx,
                "prev_id": chunks[-1]["chunk_id"] if len(chunks) > 0 else None,
                "next_id": None,
                "parent_chunk_id": None,
            }))
            last_chunk_len = 0
        elif(is_bullet_point(para)):
            if(len(chunks) == 0):
                chunks.append(create_new_chunk("child", get_bullet_point(para), {
                    "doc_id": s["wikipedia_id"],
                    "section_path": [s["wikipedia_title"]],
                    "title": s["wikipedia_title"],
                    "source_url": s["history"]["url"],
                    "paragraph_start": idx,
                    "paragraph_end": idx,
                    "prev_id": chunks[-1]["chunk_id"] if len(chunks) > 0 else None,
                    "next_id": None,
                    "parent_id": None,
                }))
            else:
                chunks[-1]["text"] += f"\n{get_bullet_point(para)}"
                chunks[-1]["paragraph_end"] = idx
            last_chunk_len += 1
        else:
            if(len(chunks) == 0):
                chunks.append(create_new_chunk("child", para.strip(), {
                    "doc_id": s["wikipedia_id"],
                    "section_path": [s["wikipedia_title"]],
                    "title": s["wikipedia_title"],
                    "source_url": s["history"]["url"],
                    "paragraph_start": idx,
                    "paragraph_end": idx,
                    "prev_id": chunks[-1]["chunk_id"] if len(chunks) > 0 else None,
                    "next_id": None,
                    "parent_id": None,
                }))
            else:
                chunks[-1]["text"] += f"\n{para.strip()}"
                chunks[-1]["paragraph_end"] = idx
            last_chunk_len += 1

        last_chunk = chunks[-1] if len(chunks) > 0 else None
        last_section = sections[-1] if len(sections) > 0 else None
        if(last_chunk and last_chunk["section_path"] is not None):
            if(last_section is None or last_chunk["section_path"][0] != last_section["section_path"][0]):
                sections.append(create_new_chunk("section", last_chunk["text"], {
                    "doc_id": s["wikipedia_id"],
                    "section_path": last_chunk["section_path"],
                    "title": s["wikipedia_title"],
                    "source_url": s["history"]["url"],
                    "paragraph_start": idx,
                    "paragraph_end": idx,
                    "prev_id": sections[-1]["chunk_id"] if len(sections) > 0 else None,
                    "next_id": None,
                    "parent_id": None,
                }))
                last_chunk["parent_id"] = sections[-1]["chunk_id"]
                sections[-1]["children_ids"].append(last_chunk["chunk_id"])
                current_section = last_chunk["section_path"][0]
            else:
                sections[-1]["text"] += f"\n{last_chunk['text']}"
                sections[-1]["paragraph_end"] = idx
                last_chunk["parent_id"] = sections[-1]["chunk_id"]
                sections[-1]["children_ids"].append(last_chunk["chunk_id"])
    print(sections)
    pages.append(create_new_chunk("page", "\n".join([s["text"] for s in sections]), {
        "doc_id": s["wikipedia_id"],
        "section_path": [s["wikipedia_title"]],
        "title": s["wikipedia_title"],
        "source_url": s["history"]["url"],
        "paragraph_start": None,
        "paragraph_end": None,
        "prev_id": pages[-1]["chunk_id"] if len(pages) > 0 else None,
        "next_id": None,
        "parent_chunk_id": None,
        "children_ids": [s["chunk_id"] for s in sections]
    }))
    if (len(pages) > 1):
        pages[-2]["next_id"] = pages[-1]["chunk_id"]

# def build_indexes(chunks):
#     # 1. prepare texts + ids + metadata
#     # 2. build sparse index
#     # 3. build dense index
#     # 4. persist both
#     return sparse_index, dense_index, chunk_store

















































































































































































































































































































































































