#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_flow_normalize.py — Chunking & labeling for RAG index

依照你的 02 規格：
- 段落/說明：300–500 tokens，overlap 20–30%
- 流程節點/表格列：80–200 tokens，overlap 0–10%
- 每個 chunk 必須包含出處（頁碼/節點/表格列）
- 讀取 abbreviations 對照（data/dicts/abbr_zh2en.json）以擴充 labels

輸入來源（預設路徑）：
- data/flows/page_XX.nodes.json（流程）
- data/flows/page_XX.table_nodes.json（表格）
- data/extracted/page_XX.json（一般段落 blocks）

輸出：
- data/chunks/chunks.jsonl（每行一個 JSON chunk）

相依：
- 可選 tiktoken（若安裝則用 GPT-3.5/4 tokenizer，否則使用簡易 token 估算器）

用法：
python src/02_flow_normalize_chunk_label.py   --flows data/flows   --extracted data/extracted   --dict data/dicts/abbr_zh2en.json   --out data/chunks/chunks.jsonl   --para-min 300 --para-max 500 --para-overlap 0.25   --node-min 80  --node-max 200 --node-overlap 0.05

"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Iterable, Optional

# =============== 可選 tokenizer：tiktoken ===============
TOK = None
try:
    import tiktoken  # type: ignore
    TOK = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOK = None


def tokenize(text: str) -> List[int]:
    if TOK is not None:
        try:
            return TOK.encode(text)
        except Exception:
            pass
    # 簡易替代：把英文單字/數字/標點視為近似 token
    return re.findall(r"\w+|\S", text)


def token_count(text: str) -> int:
    return len(tokenize(text))


def detokenize(tokens: List[str]) -> str:
    # 僅供 fallback/不會用到（寫 chunk 用原文）
    return "".join(tokens)

# =============== 段落合併與切片 ===============

def chunk_text(text: str, tmin: int, tmax: int, overlap: float) -> List[str]:
    """根據 token 長度切片。overlap 以比例表示（0.2 = 20%）。"""
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return []
    # 如果整段太短，直接回傳
    if n <= tmax:
        return [text]
    win = max(tmin, min(tmax, int(tmax)))
    step = max(1, int(win * (1 - overlap)))
    chunks = []
    i = 0
    while i < n:
        j = min(n, i + win)
        # 盡量不在句中斷：往左/右找標點分界
        sub = toks[i:j]
        s = "".join(sub) if TOK else "".join(sub)
        # 若使用 fallback tokens，直接從原文抓近似範圍
        if TOK is None:
            # 用字數比例粗抓原文片段
            frac_i = i / n
            frac_j = j / n
            start = int(frac_i * len(text))
            end = int(frac_j * len(text))
            s = text[start:end]
        chunks.append(s if s else text)
        if j == n:
            break
        i += step
    return chunks


def join_blocks_as_paragraph(blocks: List[Dict]) -> str:
    # 盡量保留段落界：heading 前後加換行
    parts = []
    for b in blocks:
        t = b.get("text", "").strip()
        if not t:
            continue
        if b.get("type") == "heading":
            parts.append("\n" + t + "\n")
        else:
            parts.append(t)
    return "\n".join(parts).strip()

# =============== labels 與詞彙對照 ===============

DEFAULT_KWS = [
    # 常見縮寫/關鍵字
    "ER", "PR", "HER2", "TNBC", "Ki-67", "SLNB", "ALND", "BCS", "RT", "NAC", "adjuvant",
    "neoadjuvant", "metastatic", "visceral crisis", "BIRADS", "CNB", "FNAB", "N0", "N1", "N2", "N3",
]


def load_abbr_dict(path: Path) -> Dict[str, Iterable[str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # 正規化：確保 value 皆為 list[str]
        out = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[k] = v
            elif isinstance(v, str):
                out[k] = [v]
        return out
    except Exception:
        return {}


def extract_labels(text: str, abbr_map: Dict[str, Iterable[str]]) -> List[str]:
    labels = set()
    # 1) 從英文關鍵詞抓
    for kw in DEFAULT_KWS:
        if re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE):
            labels.add(kw)
    # 2) 從中文對照轉英
    for zh, ens in abbr_map.items():
        if zh in text:
            for en in ens:
                labels.add(en)
    # 3) 粗略抓 regimen/劑量樣式（如 q3w, mg/m2, ×6）
    if re.search(r"q\d+w", text, re.I):
        labels.add("q#w")
    if re.search(r"mg/?m\^?2|mg/m2", text, re.I):
        labels.add("dose")
    if re.search(r"×\d+|x\d+", text, re.I):
        labels.add("cycles")
    return sorted(labels)

# =============== 載入資料源 ===============

def load_flow_nodes(flow_dir: Path) -> List[Dict]:
    nodes = []
    for p in sorted(flow_dir.glob("page_*.nodes.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            page = int(data.get("page") or re.findall(r"page_(\d+)", p.name)[0])
            for n in data.get("nodes", []):
                text = " ".join(t for t in [n.get("condition", ""), n.get("action", "")] if t).strip()
                if not text:
                    continue
                nodes.append({
                    "chunk_id": n.get("node_id", f"{p.stem}_auto"),
                    "text": text,
                    "page": page,
                    "section": None,
                    "labels": [],
                    "source": n.get("source", f"P {page:02d} flow"),
                    "lang": "mixed",
                    "node_id": n.get("node_id"),
                })
        except Exception:
            continue
    # 表格節點
    for p in sorted(flow_dir.glob("page_*.table_nodes.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            page = int(data.get("page") or re.findall(r"page_(\d+)", p.name)[0])
            for n in data.get("nodes", []):
                text = " ".join(t for t in [n.get("condition", ""), n.get("action", "")] if t).strip()
                if not text:
                    continue
                nid = n.get("node_id") or f"{p.stem}_auto"
                nodes.append({
                    "chunk_id": nid,
                    "text": text,
                    "page": page,
                    "section": None,
                    "labels": [],
                    "source": n.get("source", f"P {page:02d} table"),
                    "lang": "mixed",
                    "node_id": nid,
                })
        except Exception:
            continue
    return nodes


def load_extracted_paragraphs(ex_dir: Path) -> List[Dict]:
    paras = []
    for p in sorted(ex_dir.glob("page_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            page = int(data.get("page") or re.findall(r"page_(\d+)", p.name)[0])
            blocks = data.get("blocks", [])
            if not blocks:
                continue
            text = join_blocks_as_paragraph(blocks)
            if not text:
                continue
            # 嘗試抓頁面第一個 heading 當 section
            section = None
            for b in blocks:
                if b.get("type") == "heading":
                    section = b.get("text")
                    break
            paras.append({
                "page": page,
                "section": section,
                "text": text,
            })
        except Exception:
            continue
    return paras

# =============== 主流程 ===============

def write_jsonl(path: Path, rows: Iterable[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_chunks(
    flow_dir: Path,
    ex_dir: Path,
    dict_path: Optional[Path],
    out_path: Path,
    para_min: int, para_max: int, para_overlap: float,
    node_min: int, node_max: int, node_overlap: float,
):
    abbr_map = load_abbr_dict(dict_path) if dict_path else {}

    rows: List[Dict] = []

    # 1) 先處理流程/表格節點（通常較短）
    nodes = load_flow_nodes(flow_dir)
    for n in nodes:
        # 依長度再做細切（node 80–200 tokens）
        pieces = chunk_text(n["text"], node_min, node_max, node_overlap)
        for i, piece in enumerate(pieces, 1):
            chunk_id = n["chunk_id"] if len(pieces) == 1 else f"{n['chunk_id']}_p{i:02d}"
            labels = extract_labels(piece, abbr_map)
            rows.append({
                "chunk_id": chunk_id,
                "text": piece.strip(),
                "page": n["page"],
                "section": n["section"],
                "labels": labels,
                "source": n["source"],
                "lang": "mixed",
                "node_id": n.get("node_id"),
            })

    # 2) 再處理一般段落（300–500 tokens）
    paras = load_extracted_paragraphs(ex_dir)
    for p in paras:
        pieces = chunk_text(p["text"], para_min, para_max, para_overlap)
        for i, piece in enumerate(pieces, 1):
            chunk_id = f"p{p['page']:02d}_para_{i:03d}"
            labels = extract_labels(piece, abbr_map)
            rows.append({
                "chunk_id": chunk_id,
                "text": piece.strip(),
                "page": p["page"],
                "section": p.get("section"),
                "labels": labels,
                "source": f"P {p['page']:02d} paragraphs",
                "lang": "mixed",
                "node_id": None,
            })

    # 3) 寫出 JSONL
    write_jsonl(out_path, rows)
    print(f"✓ wrote {len(rows)} chunks -> {out_path}")


def parse_args():
    ap = argparse.ArgumentParser("02_flow_normalize: chunk & label")
    ap.add_argument("--flows", default="data/flows")
    ap.add_argument("--extracted", default="data/extracted")
    ap.add_argument("--dict", default="data/dicts/abbr_zh2en.json")
    ap.add_argument("--out", default="data/chunks/chunks.jsonl")
    ap.add_argument("--para-min", type=int, default=300)
    ap.add_argument("--para-max", type=int, default=500)
    ap.add_argument("--para-overlap", type=float, default=0.25)
    ap.add_argument("--node-min", type=int, default=80)
    ap.add_argument("--node-max", type=int, default=200)
    ap.add_argument("--node-overlap", type=float, default=0.05)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_chunks(
        flow_dir=Path(args.flows),
        ex_dir=Path(args.extracted),
        dict_path=Path(args.dict) if args.dict else None,
        out_path=Path(args.out),
        para_min=args.para_min, para_max=args.para_max, para_overlap=args.para_overlap,
        node_min=args.node_min, node_max=args.node_max, node_overlap=args.node_overlap,
    )
