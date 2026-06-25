#!/usr/bin/env python3
"""
RAG Pipeline Benchmark Script

Benchmarks every layer of the RAG pipeline — individually and combined — measuring
latency, throughput, and retrieval quality metrics.

Usage:
    python benchmark.py --layers all
    python benchmark.py --layers sparse,dense,rrf --num-queries 500
    python benchmark.py --layers chunking,sparse-index --chunking-pages 1000
"""

import argparse
import json
import math
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from db import ChunkStoreDB
from indexing import (
    SparseRetriever,
    DenseRetriever,
    build_sparse_indexes_from_db,
    build_dense_index_from_db,
)
from retrieval import hybrid_retrieve, hybrid_retrieve_with_rerank, reciprocal_rank_fusion
from reranking import Reranker, assemble_section_context
from generation import AnswerGenerator, build_context_blocks
from chunking import parse_section, parse_bullet, create_new_chunk


def latency_stats(timings_ms):
    if not timings_ms:
        return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
    s = sorted(timings_ms)
    n = len(s)
    return {
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "mean": round(statistics.mean(s), 2),
        "p50": round(s[int(n * 0.50)], 2),
        "p95": round(s[int(n * 0.95)], 2),
        "p99": round(s[int(n * 0.99)], 2),
        "count": n,
    }


def calc_qps(count, seconds):
    return round(count / seconds, 2) if seconds > 0 else 0.0


def _get_ids(results, k=None):
    ids = []
    for r in (results[:k] if k else results):
        cid = r.get("chunk_id")
        if cid:
            ids.append(cid)
    return ids


def recall_at_k(retrieved_ids, ground_truth, k):
    if not ground_truth:
        return 1.0
    gt = set(ground_truth)
    top = set(retrieved_ids[:k])
    if not top:
        return 0.0
    return len(gt & top) / len(gt)


def precision_at_k(retrieved_ids, ground_truth, k):
    if k == 0 or not retrieved_ids:
        return 0.0
    gt = set(ground_truth)
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return len([x for x in top if x in gt]) / len(top)


def compute_mrr(results_list, ground_truths, k=10):
    reciprocals = []
    for retrieved, gt in zip(results_list, ground_truths):
        if not retrieved:
            reciprocals.append(0.0)
            continue
        gt_set = set(gt)
        for rank, r in enumerate(retrieved[:k], start=1):
            cid = r.get("chunk_id")
            if cid and cid in gt_set:
                reciprocals.append(1.0 / rank)
                break
        else:
            reciprocals.append(0.0)
    return round(statistics.mean(reciprocals), 4) if reciprocals else 0.0


def compute_ndcg(results_list, ground_truths, k=10):
    ndcg_scores = []
    for retrieved, gt in zip(results_list, ground_truths):
        gt_set = set(gt)
        dcg = 0.0
        for i, r in enumerate(retrieved[:k], start=1):
            cid = r.get("chunk_id")
            rel = 1.0 if (cid and cid in gt_set) else 0.0
            dcg += rel / math.log2(i + 1)
        ideal = min(len(gt_set), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return round(statistics.mean(ndcg_scores), 4) if ndcg_scores else 0.0


def compute_quality(results_list, ground_truths, ks=(3, 5, 10)):
    metrics = {}
    for k in ks:
        recalls = [recall_at_k(_get_ids(r), gt, k) for r, gt in zip(results_list, ground_truths)]
        precs = [precision_at_k(_get_ids(r), gt, k) for r, gt in zip(results_list, ground_truths)]
        metrics[f"recall@{k}"] = round(statistics.mean(recalls), 4)
        metrics[f"precision@{k}"] = round(statistics.mean(precs), 4)
    metrics["mrr"] = compute_mrr(results_list, ground_truths, max(ks))
    metrics["ndcg@10"] = compute_ndcg(results_list, ground_truths, 10)
    return metrics


class QueryDataset:
    def __init__(self, db_path, cache_path="data/benchmark_queries.json"):
        self.db_path = db_path
        self.cache_path = cache_path

    def generate(self, num_pages=20000, num_queries=1000, seed=42):
        if Path(self.cache_path).exists():
            print(f"Loading cached queries from {self.cache_path}")
            with open(self.cache_path) as f:
                queries = json.load(f)
            if len(queries) >= num_queries:
                return queries[:num_queries]
            print(f"  cached ({len(queries)}) insufficient, regenerating...")

        print(f"Generating queries from {num_pages} Wikipedia pages...")
        db = ChunkStoreDB(self.db_path)

        ds = load_dataset(
            "facebook/kilt_wikipedia", split="full",
            trust_remote_code=True, streaming=True,
        )

        queries = []
        page_count = 0
        target_pool = num_queries * 3

        for page in ds:
            if page_count >= num_pages:
                break
            page_count += 1

            if page_count % 500 == 0:
                print(f"  scanned {page_count} pages, {len(queries)} valid queries...")

            doc_id = str(page["wikipedia_id"])
            sections = db.get_chunks_by_doc_id(doc_id, "section")

            for section in sections:
                children_ids = section.get("children_ids", [])
                if not children_ids:
                    continue
                text = section.get("text", "")
                title = text.split("\n")[0].strip() if text else ""
                if not title or len(title) < 3:
                    continue
                queries.append({
                    "query": title,
                    "doc_id": doc_id,
                    "ground_truth": children_ids,
                })

            if len(queries) >= target_pool:
                break

        db.close()
        print(f"Generated {len(queries)} candidate queries from {page_count} pages")

        random.seed(seed)
        if len(queries) > num_queries:
            queries = random.sample(queries, num_queries)

        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(queries, f, indent=2)
        print(f"Cached {len(queries)} queries to {self.cache_path}")
        return queries


def chunk_page(page_data, prev_child_id, prev_section_id):
    s = page_data
    chunks = []
    sections = []
    last_chunk_len = 0

    def _prev(items):
        return items[-1]["chunk_id"] if items else prev_child_id

    def _sprev(items):
        return items[-1]["chunk_id"] if items else prev_section_id

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
                prev_id=_prev(chunks),
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
                    prev_id=_prev(chunks),
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
                    prev_id=_prev(chunks),
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
                    prev_id=_sprev(sections),
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
        "page", "\n".join(sc["text"] for sc in sections),
        doc_id=s["wikipedia_id"],
        section_path=[s["wikipedia_title"]],
        title=s["wikipedia_title"],
        source_url=s["history"]["url"],
        paragraph_start=None,
        paragraph_end=None,
        prev_id=prev_section_id,
        next_id=None,
        parent_chunk_id=None,
        children_ids=[sc["chunk_id"] for sc in sections],
    )
    return chunks, sections, page_chunk


def _cleanup_path(path):
    for suffix in ["", "-shm", "-wal"]:
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()


class BenchmarkRunner:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self._sparse = None
        self._dense = None
        self._db = None
        self._reranker = None
        self._generator = None

    def _load_retrievers(self):
        if self._db is not None:
            return
        self._db = ChunkStoreDB(self.config["db_path"])

        sparse_shards = Path(self.config.get("sparse_shards_dir", "data/sparse_shards"))
        sparse_index = Path(self.config.get("sparse_index_path", "data/sparse_index.pkl"))

        if sparse_shards.exists() and list(sparse_shards.glob("shard_*.pkl")):
            self._sparse = SparseRetriever.load_sharded(str(sparse_shards))
            print(f"Sparse: sharded ({len(self._sparse.shards)} shards)")
        elif sparse_index.exists():
            self._sparse = SparseRetriever.load(str(sparse_index))
            n_children = len(self._sparse.chunk_store)
            db_total = self._db.count_children("child")
            print(f"Sparse: single index ({n_children} children, DB has {db_total})")
            if n_children < 1000 and db_total > 1000:
                print("  WARNING: sparse index has very few children vs DB. "
                      "Build sharded index for accurate benchmarks.")
        else:
            raise FileNotFoundError("No sparse index found")

        self._dense = DenseRetriever.load(
            storage_path=self.config.get("dense_path", "data/qdrant")
        )
        print("Dense: loaded")

    def _load_reranker(self):
        if self._reranker is None:
            self._reranker = Reranker()

    def _load_generator(self):
        if self._generator is None:
            self._generator = AnswerGenerator({
                "model": self.config.get("gen_model", "gemma-4-31B-it"),
                "temperature": self.config.get("gen_temperature", 0.2),
            })

    def _cleanup(self):
        if self._dense and hasattr(self._dense, "client"):
            self._dense.client.close()
        if self._db:
            self._db.close()

    def warmup(self, queries, n=10):
        warmup_qs = queries[:min(n, len(queries))]
        print(f"\nWarming up with {len(warmup_qs)} queries...")
        self._load_retrievers()
        self._load_reranker()

        for q in warmup_qs:
            qt = q["query"]
            self._sparse.search(qt, top_k=30)
            self._dense.search(qt, top_k=30)

        for q in warmup_qs[:3]:
            hybrid_retrieve_with_rerank(
                q["query"], self._sparse, self._dense, self._db,
                fusion_top_k=50, rerank_top_k=8, section_top_k=3,
                sparse_k=50, dense_k=50,
            )
        print("  warmup complete.")

    # --- query-layer benchmarks ---

    def benchmark_sparse(self, queries):
        self._load_retrievers()
        print("\n--- Sparse Retrieval (BM25) ---")
        timings = []
        all_results = []
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            results = self._sparse.search(q["query"], top_k=30)
            timings.append((time.perf_counter() - t0) * 1000)
            all_results.append(results)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(queries)}")
        total_s = sum(timings) / 1000
        self.results["sparse"] = {
            "latency_ms": latency_stats(timings),
            "qps": calc_qps(len(queries), total_s),
            "total_time_s": round(total_s, 2),
            "quality": compute_quality(all_results, [q["ground_truth"] for q in queries]),
        }
        self._print_layer("sparse")

    def benchmark_dense(self, queries):
        self._load_retrievers()
        print("\n--- Dense Retrieval (fastembed + Qdrant) ---")
        timings = []
        embed_ms = []
        search_ms = []
        all_results = []
        for i, q in enumerate(queries):
            qt = q["query"]
            t0 = time.perf_counter()
            emb = list(self._dense.model.embed([qt]))[0]
            et = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            qr = self._dense.client.query_points(
                collection_name=self._dense.collection_name,
                query=emb.tolist(), limit=30,
            )
            st = (time.perf_counter() - t1) * 1000
            hits = []
            for pt in qr.points:
                c = dict(pt.payload)
                c["score"] = pt.score
                hits.append(c)
            timings.append(et + st)
            embed_ms.append(et)
            search_ms.append(st)
            all_results.append(hits)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(queries)}")
        total_s = sum(timings) / 1000
        self.results["dense"] = {
            "latency_ms": latency_stats(timings),
            "qps": calc_qps(len(queries), total_s),
            "total_time_s": round(total_s, 2),
            "embed_latency_ms": latency_stats(embed_ms),
            "search_latency_ms": latency_stats(search_ms),
            "quality": compute_quality(all_results, [q["ground_truth"] for q in queries]),
        }
        self._print_layer("dense")

    def benchmark_rrf(self, queries):
        self._load_retrievers()
        print("\n--- RRF Fusion ---")
        print("  pre-computing sparse+dense results...")
        sparse_r = []
        dense_r = []
        for i, q in enumerate(queries):
            qt = q["query"]
            sparse_r.append(self._sparse.search(qt, top_k=50))
            dense_r.append(self._dense.search(qt, top_k=50))
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(queries)}")
        timings = []
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            fused = reciprocal_rank_fusion([sparse_r[i], dense_r[i]], k=60)
            sorted_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)
            results = []
            seen = set()
            for cid in sorted_ids:
                if cid in seen:
                    continue
                seen.add(cid)
                chunk = self._db.get_chunk(cid)
                if chunk is None:
                    continue
                results.append({**chunk, "score": fused[cid]})
                if len(results) >= 10:
                    break
            timings.append((time.perf_counter() - t0) * 1000)
        total_s = sum(timings) / 1000
        self.results["rrf"] = {
            "latency_ms": latency_stats(timings),
            "qps": calc_qps(len(queries), total_s),
            "total_time_s": round(total_s, 2),
        }
        self._print_layer("rrf")

    def benchmark_rerank(self, queries):
        self._load_retrievers()
        self._load_reranker()
        print("\n--- Reranking (Cross-Encoder) ---")
        print("  pre-computing fusion candidates...")
        candidates_list = []
        for i, q in enumerate(queries):
            fr = hybrid_retrieve(
                q["query"], self._sparse, self._dense, self._db,
                top_k=50, sparse_k=50, dense_k=50, expand_to_section=False,
            )
            candidates_list.append(fr["results"])
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{len(queries)}")
        timings = []
        all_reranked = []
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            rr = self._reranker.rerank(q["query"], candidates_list[i], top_k=8)
            timings.append((time.perf_counter() - t0) * 1000)
            all_reranked.append(rr)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(queries)}")
        total_s = sum(timings) / 1000
        self.results["rerank"] = {
            "latency_ms": latency_stats(timings),
            "qps": calc_qps(len(queries), total_s),
            "total_time_s": round(total_s, 2),
            "quality": compute_quality(all_reranked, [q["ground_truth"] for q in queries]),
        }
        self._print_layer("rerank")

    def benchmark_section_assembly(self, queries):
        self._load_retrievers()
        self._load_reranker()
        print("\n--- Section Assembly (DB lookups) ---")
        print("  pre-computing reranked children...")
        reranked_list = []
        for i, q in enumerate(queries):
            fr = hybrid_retrieve(
                q["query"], self._sparse, self._dense, self._db,
                top_k=50, sparse_k=50, dense_k=50, expand_to_section=False,
            )
            rr = self._reranker.rerank(q["query"], fr["results"], top_k=8)
            reranked_list.append(rr)
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{len(queries)}")
        timings = []
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            assemble_section_context(reranked_list[i], self._db, top_sections=3)
            timings.append((time.perf_counter() - t0) * 1000)
        total_s = sum(timings) / 1000
        self.results["section_assembly"] = {
            "latency_ms": latency_stats(timings),
            "qps": calc_qps(len(queries), total_s),
            "total_time_s": round(total_s, 2),
        }
        self._print_layer("section_assembly")

    def benchmark_combined(self, queries):
        self._load_retrievers()
        self._load_reranker()
        print("\n--- Combined Retrieval (sparse -> dense -> RRF -> rerank -> assembly) ---")
        timings = []
        all_child_results = []
        for i, q in enumerate(queries):
            qt = q["query"]
            t0 = time.perf_counter()
            result = hybrid_retrieve_with_rerank(
                qt, self._sparse, self._dense, self._db,
                fusion_top_k=50, rerank_top_k=8, section_top_k=3,
                sparse_k=50, dense_k=50,
            )
            timings.append((time.perf_counter() - t0) * 1000)
            child_ids = []
            for sec in result["results"]:
                child_ids.extend(sec.get("child_ids", []))
            all_child_results.append([{"chunk_id": cid, "score": 1.0} for cid in child_ids])
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(queries)}")
        total_s = sum(timings) / 1000
        self.results["combined_retrieval"] = {
            "latency_ms": latency_stats(timings),
            "qps": calc_qps(len(queries), total_s),
            "total_time_s": round(total_s, 2),
            "quality": compute_quality(all_child_results, [q["ground_truth"] for q in queries]),
        }
        self._print_layer("combined_retrieval")

    def benchmark_generation(self, queries):
        self._load_retrievers()
        self._load_reranker()
        self._load_generator()
        gen_count = self.config.get("gen_queries", 50)
        gen_qs = queries[:gen_count]
        print(f"\n--- LLM Generation ({len(gen_qs)} queries) ---")
        print("  pre-building context blocks...")
        blocks_list = []
        for i, q in enumerate(gen_qs):
            result = hybrid_retrieve_with_rerank(
                q["query"], self._sparse, self._dense, self._db,
            )
            blocks_list.append(build_context_blocks(result["results"]))
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(gen_qs)}")
        timings = []
        tok_counts = []
        grounded = 0
        total_tok = 0
        for i, q in enumerate(gen_qs):
            t0 = time.perf_counter()
            answer = self._generator.generate(q["query"], blocks_list[i])
            ms = (time.perf_counter() - t0) * 1000
            timings.append(ms)
            if answer.get("answer_text"):
                tc = len(answer["answer_text"].split())
                tok_counts.append(tc)
                total_tok += tc
            if answer.get("grounded"):
                grounded += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(gen_qs)}")
        total_s = sum(timings) / 1000
        self.results["generation"] = {
            "latency_ms": latency_stats(timings),
            "qps": calc_qps(len(gen_qs), total_s),
            "total_time_s": round(total_s, 2),
            "avg_output_tokens": round(statistics.mean(tok_counts), 1) if tok_counts else 0,
            "total_output_tokens": total_tok,
            "tokens_per_second": round(total_tok / total_s, 1) if total_s > 0 else 0,
            "grounded_ratio": round(grounded / len(gen_qs), 3) if gen_qs else 0,
            "num_queries": len(gen_qs),
        }
        self._print_layer("generation")

    def benchmark_generation_standalone(self, queries):
        self._load_retrievers()
        self._load_reranker()
        self._load_generator()
        gen_count = self.config.get("gen_queries", 50)
        gen_qs = queries[:gen_count]
        print(f"\n--- Generation Standalone ({len(gen_qs)} queries) ---")
        print("  pre-building context blocks...")
        blocks_list = []
        for i, q in enumerate(gen_qs):
            result = hybrid_retrieve_with_rerank(
                q["query"], self._sparse, self._dense, self._db,
            )
            blocks_list.append(build_context_blocks(result["results"]))
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(gen_qs)}")
        timings = []
        tok_counts = []
        total_tok = 0
        for i, q in enumerate(gen_qs):
            t0 = time.perf_counter()
            answer = self._generator.generate(q["query"], blocks_list[i])
            ms = (time.perf_counter() - t0) * 1000
            timings.append(ms)
            if answer.get("answer_text"):
                tc = len(answer["answer_text"].split())
                tok_counts.append(tc)
                total_tok += tc
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(gen_qs)}")
        total_s = sum(timings) / 1000
        self.results["generation_standalone"] = {
            "latency_ms": latency_stats(timings),
            "qps": calc_qps(len(gen_qs), total_s),
            "total_time_s": round(total_s, 2),
            "avg_output_tokens": round(statistics.mean(tok_counts), 1) if tok_counts else 0,
            "total_output_tokens": total_tok,
            "tokens_per_second": round(total_tok / total_s, 1) if total_s > 0 else 0,
            "num_queries": len(gen_qs),
        }
        self._print_layer("generation_standalone")

    # --- batch-layer benchmarks ---

    def benchmark_chunking(self):
        num_pages = self.config.get("query_source_pages", 500)
        temp_db = Path("data/benchmark_chunks.db")
        print(f"\n--- Chunking Benchmark ({num_pages} pages) ---")
        _cleanup_path(temp_db)

        tdb = ChunkStoreDB(str(temp_db))
        ds = load_dataset(
            "facebook/kilt_wikipedia", split="full",
            trust_remote_code=True, streaming=True,
        )
        total_paras = 0
        total_children = 0
        total_sections = 0
        prev_cid = None
        prev_sid = None

        t0 = time.perf_counter()
        for i, page in enumerate(ds):
            if i >= num_pages:
                break
            children, sections, page_chunk = chunk_page(page, prev_cid, prev_sid)
            for c in children:
                tdb.insert_chunk(c)
            for sc in sections:
                tdb.insert_chunk(sc)
            tdb.insert_chunk(page_chunk)
            if children:
                prev_cid = children[-1]["chunk_id"]
            if sections:
                prev_sid = sections[-1]["chunk_id"]
            total_paras += len(page["text"]["paragraph"])
            total_children += len(children)
            total_sections += len(sections)
            if (i + 1) % 100 == 0:
                tdb.commit()
                print(f"  {i + 1}/{num_pages}")
        tdb.commit()
        total_s = time.perf_counter() - t0
        tdb.close()

        self.results["chunking"] = {
            "total_time_s": round(total_s, 2),
            "num_pages": num_pages,
            "pages_per_sec": round(num_pages / total_s, 2) if total_s > 0 else 0,
            "paragraphs_per_sec": round(total_paras / total_s, 1) if total_s > 0 else 0,
            "total_paragraphs": total_paras,
            "total_children": total_children,
            "total_sections": total_sections,
            "children_per_sec": round(total_children / total_s, 1) if total_s > 0 else 0,
        }
        self._print_batch("chunking")

        if not self.config.get("keep_temp"):
            _cleanup_path(temp_db)

    def benchmark_sparse_indexing(self):
        shard_size = self.config.get("sparse_shard_size", 50000)
        temp_db = Path("data/benchmark_chunks.db")
        temp_dir = Path("data/benchmark_sparse_shards")
        print(f"\n--- Sparse Indexing Benchmark (shard_size={shard_size}) ---")

        if not temp_db.exists():
            print("  SKIPPED: temp DB not found (run chunking first)")
            self.results["sparse_indexing"] = {"error": "temp DB not found"}
            return

        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        temp_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        build_sparse_indexes_from_db(str(temp_db), str(temp_dir), shard_size=shard_size)
        total_s = time.perf_counter() - t0

        shard_files = sorted(temp_dir.glob("shard_*.pkl"))
        import pickle
        total_children = 0
        for sf in shard_files:
            with open(sf, "rb") as f:
                data = pickle.load(f)
            total_children += len(data.get("chunk_store", []))

        self.results["sparse_indexing"] = {
            "total_time_s": round(total_s, 2),
            "num_shards": len(shard_files),
            "total_children": total_children,
            "children_per_sec": round(total_children / total_s, 1) if total_s > 0 else 0,
            "shard_size": shard_size,
        }
        self._print_batch("sparse_indexing")

        if not self.config.get("keep_temp"):
            shutil.rmtree(str(temp_dir), ignore_errors=True)

    def benchmark_dense_indexing(self):
        batch_size = self.config.get("dense_batch_size", 1000)
        temp_db = Path("data/benchmark_chunks.db")
        temp_qdrant = Path("data/benchmark_qdrant")
        print(f"\n--- Dense Indexing Benchmark (batch_size={batch_size}) ---")

        if not temp_db.exists():
            print("  SKIPPED: temp DB not found (run chunking first)")
            self.results["dense_indexing"] = {"error": "temp DB not found"}
            return

        if temp_qdrant.exists():
            shutil.rmtree(str(temp_qdrant))

        t0 = time.perf_counter()
        build_dense_index_from_db(str(temp_db), str(temp_qdrant), batch_size=batch_size)
        total_s = time.perf_counter() - t0

        tdb = ChunkStoreDB(str(temp_db))
        total_children = tdb.count_children("child")
        tdb.close()

        self.results["dense_indexing"] = {
            "total_time_s": round(total_s, 2),
            "total_children": total_children,
            "children_per_sec": round(total_children / total_s, 1) if total_s > 0 else 0,
            "batch_size": batch_size,
        }
        self._print_batch("dense_indexing")

        if not self.config.get("keep_temp"):
            shutil.rmtree(str(temp_qdrant), ignore_errors=True)

    # --- helpers ---

    def _print_layer(self, name):
        r = self.results.get(name, {})
        lat = r.get("latency_ms", {})
        print(f"  latency  mean={lat.get('mean', '-')}ms  p50={lat.get('p50', '-')}ms  "
              f"p95={lat.get('p95', '-')}ms  p99={lat.get('p99', '-')}ms")
        print(f"  QPS: {r.get('qps', '-')}")
        q = r.get("quality")
        if q:
            parts = []
            for k in ["recall@5", "recall@10", "mrr", "ndcg@10"]:
                if k in q:
                    parts.append(f"{k}={q[k]:.4f}")
            print(f"  quality: {'  '.join(parts)}")
        if "avg_output_tokens" in r:
            print(f"  tokens: {r['avg_output_tokens']}/q avg, {r['tokens_per_second']}/s, "
                  f"grounded={r.get('grounded_ratio', '-')}")

    def _print_batch(self, name):
        r = self.results.get(name, {})
        if "error" in r:
            print(f"  SKIPPED: {r['error']}")
            return
        print(f"  time: {r.get('total_time_s', '-')}s")
        if "pages_per_sec" in r:
            print(f"  throughput: {r['pages_per_sec']} pages/s  "
                  f"{r['paragraphs_per_sec']} paras/s  {r['children_per_sec']} children/s")
        else:
            print(f"  throughput: {r.get('children_per_sec', '-')} children/s  "
                  f"({r.get('total_children', '-')} children total)")

    def run(self, queries):
        layers = self.config.get("layers", set())
        query_layer_names = [
            "sparse", "dense", "rrf", "rerank", "section_assembly",
            "combined", "generation", "generation_standalone",
        ]
        batch_layer_names = ["chunking", "sparse_indexing", "dense_indexing"]

        query_layers_requested = layers & set(query_layer_names)
        if query_layers_requested:
            warmup_n = self.config.get("warmup", 10)
            if warmup_n > 0:
                self.warmup(queries, n=warmup_n)

        for name in query_layer_names:
            if name in layers:
                getattr(self, f"benchmark_{name}")(queries)

        for name in batch_layer_names:
            if name in layers:
                getattr(self, f"benchmark_{name}")()

        self._cleanup()

    def print_summary(self):
        print("\n" + "=" * 95)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 95)

        qlayers = [
            "sparse", "dense", "rrf", "rerank", "section_assembly",
            "combined_retrieval", "generation", "generation_standalone",
        ]
        header = f"{'Layer':<25} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'QPS':>8}  Quality/Tokens"
        print(f"\n{header}")
        print("-" * 95)

        for name in qlayers:
            r = self.results.get(name)
            if not r:
                continue
            lat = r.get("latency_ms", {})
            q = r.get("quality", {})
            extra = ""
            if q:
                parts = []
                for k in ("mrr", "recall@10"):
                    if k in q:
                        parts.append(f"{k}={q[k]:.3f}")
                extra = "  ".join(parts)
            elif "grounded_ratio" in r:
                extra = f"grounded={r['grounded_ratio']:.2f}  {r.get('tokens_per_second', 0)}tok/s"
            elif "avg_output_tokens" in r:
                extra = f"{r['avg_output_tokens']}tok/q  {r.get('tokens_per_second', 0)}tok/s"
            print(f"{name:<25} {lat.get('mean', 0):>8.1f} {lat.get('p50', 0):>8.1f} "
                  f"{lat.get('p95', 0):>8.1f} {lat.get('p99', 0):>8.1f} "
                  f"{r.get('qps', 0):>8.1f}  {extra}")

        blayers = ["chunking", "sparse_indexing", "dense_indexing"]
        if any(n in self.results for n in blayers):
            print(f"\n{'Layer':<25} {'Time(s)':>10}  Throughput")
            print("-" * 60)
            for name in blayers:
                r = self.results.get(name)
                if not r:
                    continue
                if "error" in r:
                    print(f"{name:<25} SKIPPED: {r['error']}")
                    continue
                if "pages_per_sec" in r:
                    tp = f"{r['pages_per_sec']} pages/s  ({r['children_per_sec']} children/s)"
                else:
                    tp = f"{r['children_per_sec']} children/s"
                print(f"{name:<25} {r.get('total_time_s', 0):>10.2f}  {tp}")

        print("\n" + "=" * 95)

    def save_json(self, path):
        output = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {k: (sorted(v) if isinstance(v, set) else v) for k, v in self.config.items()},
            },
            "results": self.results,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nFull results saved to {path}")


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Benchmark RAG pipeline layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python benchmark.py --layers all
  python benchmark.py --layers sparse,dense,rrf --num-queries 500
  python benchmark.py --layers chunking,sparse_indexing,dense_indexing --query-source-pages 2000""",
    )
    parser.add_argument("--layers", default="all",
                        help="Comma-separated: sparse,dense,rrf,rerank,section_assembly,"
                             "combined,generation,generation_standalone,"
                             "chunking,sparse_indexing,dense_indexing  (or 'all')")
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--query-source-pages", type=int, default=20000,
                        help="Pages streamed from dataset (used for query generation, chunking, and indexing benchmarks)")
    parser.add_argument("--sparse-shard-size", type=int, default=50000)
    parser.add_argument("--dense-batch-size", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--gen-queries", type=int, default=50)
    parser.add_argument("--gen-model", default="gemma-4-31B-it")
    parser.add_argument("--gen-temperature", type=float, default=0.2)
    parser.add_argument("--output", default=None)
    parser.add_argument("--query-cache", default="data/benchmark_queries.json")
    parser.add_argument("--db-path", default="data/chunks.db")
    parser.add_argument("--sparse-index-path", default="data/sparse_index.pkl")
    parser.add_argument("--sparse-shards-dir", default="data/sparse_shards")
    parser.add_argument("--dense-path", default="data/qdrant")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    all_layers = {
        "sparse", "dense", "rrf", "rerank", "section_assembly",
        "combined", "generation", "generation_standalone",
        "chunking", "sparse_indexing", "dense_indexing",
    }
    if args.layers == "all":
        layers = all_layers
    else:
        layers = set(args.layers.split(","))
        invalid = layers - all_layers
        if invalid:
            print(f"Invalid layers: {invalid}")
            print(f"Valid: {', '.join(sorted(all_layers))}")
            sys.exit(1)

    config = {
        "layers": layers,
        "num_queries": args.num_queries,
        "query_source_pages": args.query_source_pages,
        "sparse_shard_size": args.sparse_shard_size,
        "dense_batch_size": args.dense_batch_size,
        "warmup": args.warmup,
        "gen_queries": args.gen_queries,
        "gen_model": args.gen_model,
        "gen_temperature": args.gen_temperature,
        "db_path": args.db_path,
        "sparse_index_path": args.sparse_index_path,
        "sparse_shards_dir": args.sparse_shards_dir,
        "dense_path": args.dense_path,
        "keep_temp": args.keep_temp,
        "seed": args.seed,
    }

    query_layers = {
        "sparse", "dense", "rrf", "rerank", "section_assembly",
        "combined", "generation", "generation_standalone",
    }
    queries = []
    if layers & query_layers:
        qds = QueryDataset(args.db_path, args.query_cache)
        queries = qds.generate(
            num_pages=args.query_source_pages,
            num_queries=args.num_queries,
            seed=args.seed,
        )
        print(f"Using {len(queries)} benchmark queries\n")

    runner = BenchmarkRunner(config)
    runner.run(queries)
    runner.print_summary()

    output_path = args.output
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/benchmark_results_{ts}.json"
    runner.save_json(output_path)


if __name__ == "__main__":
    main()
