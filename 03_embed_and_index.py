#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_embed_and_index.py — 多語嵌入 + L2 正規化 + FAISS 索引（含 e5 前綴選項）

更新：
- 新增 `--prefix {none,e5}`（預設：`e5`）。
  - `e5`：對文件加 `"passage: "`，對查詢加 `"query: "`（符合 e5 家族建議）。
  - `none`：不加前綴（MiniLM/LaBSE 等可用）。

流程：
- 讀取 data/chunks/chunks.jsonl（02 產物）
- 產生嵌入 → **L2 normalize** → 存 `index/vectors.npy`
- 建 **FAISS**：IndexFlatIP（或選 HNSWFlat），存 `index/faiss.index`
- 存 `ids.json`（chunk_id→row_id）與 `meta.jsonl`

用法：
python src/03_embed_and_index.py   --chunks data/chunks/chunks.jsonl   --model model/multilingual-e5-base   --out-dir data/index   --backend flat   --prefix e5 

查詢測試：
python src/03_embed_and_index.py --search "乳房超音波 BIRADS III 要做什麼？" \
  --topk 5 --dict data/dicts/abbr_zh2en.json \
  --model intfloat/multilingual-e5-base --prefix e5
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# sentence-transformers 會下載模型；若離線，請提供本機模型路徑
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

# FAISS（Windows 請安裝 faiss-cpu）
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

import re


def load_chunks(jsonl_path: Path) -> Tuple[List[str], List[Dict]]:
    texts: List[str] = []
    metas: List[Dict] = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get('text', '').strip()
            if not text:
                continue
            texts.append(text)
            metas.append(obj)
    return texts, metas


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def build_faiss_index(vectors: np.ndarray, backend: str = 'flat', hnsw_m: int = 32) -> 'faiss.Index':
    if faiss is None:
        raise RuntimeError('faiss 未安裝，請 pip install faiss-cpu')
    dim = vectors.shape[1]
    if backend == 'flat':
        index = faiss.IndexFlatIP(dim)  # 內積 = 餘弦（向量已 L2 normalize）
    elif backend == 'hnsw':
        index = faiss.IndexHNSWFlat(dim, hnsw_m)
        index.hnsw.efConstruction = 200
    else:
        raise ValueError('backend 僅支援 flat | hnsw')
    index.add(vectors.astype(np.float32))
    return index


def save_meta(out_dir: Path, metas: List[Dict], ids: List[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    id_map = {cid: i for i, cid in enumerate(ids)}
    (out_dir / 'ids.json').write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out_dir / 'meta.jsonl').open('w', encoding='utf-8') as f:
        for i, m in enumerate(metas):
            m2 = dict(m)
            m2['row_id'] = i
            f.write(json.dumps(m2, ensure_ascii=False) + '')


def load_abbr(path: Optional[Path]) -> Dict[str, List[str]]:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        norm = {}
        for k, v in data.items():
            norm[k] = v if isinstance(v, list) else [v]
        return norm
    except Exception:
        return {}


def expand_query(q: str, abbr: Dict[str, List[str]]) -> str:
    for zh, ens in abbr.items():
        if zh in q:
            q += ' ' + ' '.join(ens)
    return q


def apply_prefix(texts: List[str], mode: str, kind: str) -> List[str]:
    """mode: none|e5 ; kind: 'doc' or 'query'"""
    if mode == 'e5':
        pref = 'passage: ' if kind == 'doc' else 'query: '
        return [pref + t for t in texts]
    return texts


def embed_texts(model, texts: List[str], batch: int = 64, normalize: bool = True) -> np.ndarray:
    embs = model.encode(texts, batch_size=batch, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    if normalize:
        embs = l2_normalize(embs)
    return embs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser('03_embed_and_index: multilingual embeddings + FAISS index')
    ap.add_argument('--chunks', default='data/chunks/chunks.jsonl')
    ap.add_argument('--model', required=True, help='多語模型名稱或本機路徑，例如 intfloat/multilingual-e5-base 或 C:/models/e5')
    ap.add_argument('--out-dir', default='data/index')
    ap.add_argument('--backend', choices=['flat', 'hnsw'], default='flat')
    ap.add_argument('--hnsw-m', type=int, default=32)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--dict', default='')
    ap.add_argument('--prefix', choices=['none','e5'], default='e5', help='前綴策略：e5=doc→passage:, query→query:；none=不加')
    ap.add_argument('--search', default='', help='（可選）測試查詢字串；若提供則在建好索引後直接查前 topk')
    ap.add_argument('--topk', type=int, default=5)
    args = ap.parse_args()

    if SentenceTransformer is None:
        raise RuntimeError('請先安裝 sentence-transformers：pip install sentence-transformers')
    if faiss is None:
        raise RuntimeError('請先安裝 faiss-cpu：pip install faiss-cpu')

    chunks_path = Path(args.chunks)
    texts, metas = load_chunks(chunks_path)
    if not texts:
        raise SystemExit(f'找不到可用 chunks（{chunks_path} 為空？）')

    # 載入模型（可為本機資料夾）
    print(f'[model] loading: {args.model}')
    model = SentenceTransformer(args.model)

    # 文檔前綴
    texts_pref = apply_prefix(texts, args.prefix, kind='doc')

    # 產生嵌入並 L2 normalize
    print(f'[embed] encoding {len(texts_pref)} chunks ...')
    vecs = embed_texts(model, texts_pref, batch=args.batch, normalize=True)
    dim = vecs.shape[1]
    print(f'[embed] vectors shape: {vecs.shape}, dtype={vecs.dtype}')

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'vectors.npy', vecs)

    # 構建索引
    print(f'[faiss] building index backend={args.backend}')
    index = build_faiss_index(vecs, backend=args.backend, hnsw_m=args.hnsw_m)
    faiss.write_index(index, str(out_dir / 'faiss.index'))

    # 對應表與 meta
    ids = [m.get('chunk_id', str(i)) for i, m in enumerate(metas)]
    save_meta(out_dir, metas, ids)

    # 可選：查詢測試
    if args.search:
        abbr = load_abbr(Path(args.dict)) if args.dict else {}
        q = expand_query(args.search, abbr)
        q = apply_prefix([q], args.prefix, kind='query')[0]
        print(f"[search] query: {q}")
        qv = embed_texts(model, [q], batch=1, normalize=True)  # shape (1, dim)
        D, I = index.search(qv, args.topk)  # 內積 == cosine（向量已正規化）
        print('[search] topk IDs & scores:')
        for rank, (idx, score) in enumerate(zip(I[0].tolist(), D[0].tolist()), 1):
            m = metas[idx]
            print(f"#{rank}	score={score:.4f}	chunk_id={m.get('chunk_id')}	page={m.get('page')}	source={m.get('source')}")

    print(f'完成：- {out_dir / "vectors.npy"} \
          - {out_dir / "faiss.index"}\
          - {out_dir / "ids.json"} \
          - {out_dir / "meta.jsonl"}')


if __name__ == '__main__':
    main()
