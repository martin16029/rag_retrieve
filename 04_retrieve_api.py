#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_retrieve_api.py — 檢索 API（不經 LLM，直接回原文 chunks）

暴露單一入口：
    search(query_zh: str, top_n: int = 5, *,
           index_dir: str = 'data/index',
           dict_path: str = 'data/dicts/abbr_zh2en.json',
           model: str = 'intfloat/multilingual-e5-base',
           prefix: str = 'e5',              # 'e5' | 'none'（需與 03 設定一致）
           K: int = 20,                     # 稠密先取 Top-K 候選
           use_bm25: bool = False,          # 是否做 BM25 混合
           alpha: float = 0.7,              # 混合權重：alpha*cos + (1-alpha)*bm25
           use_translate: bool = False,     # （可選）把中文→英文再檢索並合併
           dedup_cos: float = 0.95,         # 向量相似去重閾值
           show_scores: bool = True) -> dict

輸出格式：
{
  "query": "...",
  "hits": [
     {"rank":1, "score":0.83, "text":"...", "page":15, "section":"...",
      "source":"...", "chunk_id":"..."}, ...
  ]
}

相依：
- numpy, faiss-cpu, sentence-transformers（離線可使用本機模型路徑）
- （可選）rank-bm25；否則 fallback 用 TF-IDF（scikit-learn）

注意：
- 需要先跑 02 與 03，已產生：
  - data/index/faiss.index
  - data/index/vectors.npy（僅供維度/調試）
  - data/index/meta.jsonl（對應 chunk 原文與屬性）

指令：
python src/04_retrieve_api.py --q "HER2陰性早期乳癌輔助化療首選？" --topn 5   --index data/index   --dict data/dicts/abbr_zh2en.json   --model model/multilingual-e5-base   --prefix e5 --K 20 --bm25  
  
  """
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# FAISS
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# Embedding
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

# （可選）BM25；若無則 TF-IDF fallback
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

# TF-IDF fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    HAS_TFIDF = True
except Exception:
    HAS_TFIDF = False

import re

# ======================= 工具 =======================

def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms

# ---------- meta.jsonl 安全讀取（含自修復） ----------
def load_meta_safe(index_dir: Path, repair: bool = True, backup: bool = True) -> Tuple[List[Dict], List[str]]:
    """
    讀取 data/index/meta.jsonl，必要時自動修復：
    - 拆開黏在一起的 JSON 物件（"}{" -> "}\n{"）
    - 過濾掉非 JSON 行（如日誌殘留）
    - 修好後會備份成 .bak.jsonl 並覆寫原檔（可關閉 backup）
    """
    metas: List[Dict] = []
    ids: List[str] = []
    meta_path = index_dir / 'meta.jsonl'
    if not meta_path.exists():
        raise FileNotFoundError(f'meta not found: {meta_path}')

    raw = meta_path.read_text(encoding='utf-8', errors='ignore')

    def try_lines(text: str) -> List[Dict]:
        out: List[Dict] = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            if not (s.startswith('{') and s.endswith('}')):
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                import re as _re
                s2 = _re.sub(r"\s+", " ", s)
                try:
                    out.append(json.loads(s2))
                except Exception:
                    continue
        return out

    ok = try_lines(raw)  # 先嘗試原始逐行解析

    if not ok and repair:
        import re as _re
        fixed = raw.replace("\ufeff", "")
        fixed = _re.sub(r"}\s*{", "}\n{", fixed)  # 拆開黏連
        ok = try_lines(fixed)
        if not ok:
            raise ValueError(f'Failed to parse meta.jsonl even after repair: {meta_path}')
        if backup:
            (meta_path.with_suffix('.bak.jsonl')).write_text(raw, encoding='utf-8')
        with meta_path.open('w', encoding='utf-8') as f:
            for i, m in enumerate(ok):
                m2 = dict(m); m2.setdefault('row_id', i)
                f.write(json.dumps(m2, ensure_ascii=False) + "\n")

    if not ok:
        raise ValueError(f'meta.jsonl contains malformed lines and repair=False: {meta_path}')

    metas = ok
    for i, m in enumerate(metas):
        ids.append(m.get('chunk_id', str(i)))
    return metas, ids

def load_meta(index_dir: Path) -> Tuple[List[Dict], List[str]]:
    metas: List[Dict] = []
    ids: List[str] = []
    meta_path = index_dir / 'meta.jsonl'
    if not meta_path.exists():
        raise FileNotFoundError(f'meta not found: {meta_path}')
    with meta_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            metas.append(m)
            ids.append(m.get('chunk_id', str(len(ids))))
    return metas, ids


def load_abbr(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        out = {}
        for k, v in data.items():
            out[k] = v if isinstance(v, list) else [v]
        return out
    except Exception:
        return {}


def expand_query_by_abbr(q: str, abbr: Dict[str, List[str]]) -> str:
    for zh, ens in abbr.items():
        if zh in q:
            q += ' ' + ' '.join(ens)
    return q


def apply_prefix(text: str, prefix: str, kind: str) -> str:
    if prefix == 'e5':
        return ('query: ' if kind == 'query' else 'passage: ') + text
    return text


def jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    def ngrams(s):
        s = re.sub(r"\s+", " ", s.strip().lower())
        return {s[i:i+n] for i in range(max(0, len(s)-n+1))}
    A, B = ngrams(a), ngrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

# ---------- 參考文獻偵測與段落類型加權 ----------
_REF_PAT = re.compile(r"(N\s*Engl\s*J\s*Med|J\s*Clin\s*Oncol|Ann\s*Oncol|Lancet|JAMA|\d{4};\d+|\(\d{4}\))", re.I)
_DEFIS = re.compile(r"\b(References?|參考文獻|參考資料)\b", re.I)

def is_bibliography(meta: Dict) -> bool:
    text = (meta.get('text') or '').strip()
    src  = (meta.get('source') or '')
    sec  = (meta.get('section') or '')
    cid  = str(meta.get('chunk_id') or '')
    # 明確標示或欄位提及
    if _DEFIS.search(text) or _DEFIS.search(src) or _DEFIS.search(sec):
        return True
    # 期刊/年份模式密集（常見於參考文獻）
    if _REF_PAT.search(text) and len(text) < 1500:
        return True
    # chunk_id 提示
    if 'ref' in cid.lower() or 'refs' in cid.lower():
        return True
    return False

def type_bonus(meta: Dict) -> float:
    """表格/流程給正向加權；偵測到參考文獻則強扣分。"""
    if is_bibliography(meta):
        return -0.5  # 直接把文獻列表壓到後面
    bonus = 0.0
    cid = (meta.get('chunk_id') or '').lower()
    src = (meta.get('source') or '').lower()
    # 表格與流程圖通常更像「可直接回答」的單位
    if 'tbl' in cid or 'table' in src:
        bonus += 0.20
    if 'node' in cid or 'flow' in src:
        bonus += 0.15
    return bonus


# ======================= 載入索引 =======================

class Retriever:
    def __init__(self, index_dir: str, model_name_or_path: str, prefix: str = 'e5', repair_meta: bool = True):
        if faiss is None:
            raise RuntimeError('需要 faiss-cpu：pip install faiss-cpu')
        if SentenceTransformer is None:
            raise RuntimeError('需要 sentence-transformers：pip install sentence-transformers')
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(model_name_or_path)
        self.prefix = prefix
        # FAISS index
        self.index = faiss.read_index(str(self.index_dir / 'faiss.index'))
        # meta（含 chunk 原文與屬性）
        self.metas, self.ids = load_meta_safe(self.index_dir, repair=repair_meta)
        # 可選：載 vectors.npy 用於維度檢查或快速相似度
        self.vecs_path = self.index_dir / 'vectors.npy'
        self.dim = None
        if self.vecs_path.exists():
            try:
                vecs = np.load(self.vecs_path)
                self.dim = vecs.shape[1]
            except Exception:
                self.dim = None

    def encode_query(self, q: str) -> np.ndarray:
        q = apply_prefix(q, self.prefix, 'query')
        v = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=False)
        v = l2_normalize(v).astype(np.float32)
        return v

# ======================= 主 API =======================

# 4.1 label/關鍵詞命中加分
def label_bonus(meta: Dict, abbr, query_zh) -> float:
    bonus = 0.0
    labels = set(meta.get('labels', []) or [])
    for zh, ens in abbr.items():
        if zh in query_zh:
            for e in ens:
                if e in labels:
                    bonus += 0.03
    # 查詢中的英文 token 命中 labels
    for tok in re.findall(r"[A-Za-z0-9+-]+", query_zh):
        if tok in labels:
            bonus += 0.03
    return min(bonus, 0.10)


def search(query_zh: str,
           top_n: int = 5,
           *,
           index_dir: str = 'data/index',
           dict_path: str = 'data/dicts/abbr_zh2en.json',
           model: str = 'intfloat/multilingual-e5-base',
           prefix: str = 'e5',
           K: int = 20,
           use_bm25: bool = False,
           alpha: float = 0.7,
           use_translate: bool = False,
           dedup_cos: float = 0.95,
           show_scores: bool = True,
           repair_meta: bool = True) -> Dict:
    """高階檢索函數：依規格做擴展→稠密→（可選 BM25）→規則重排與去重。"""
    retr = Retriever(index_dir=index_dir, model_name_or_path=model, prefix=prefix, repair_meta=repair_meta)

    # 1) 查詢前處理（中文擴展）
    abbr = load_abbr(Path(dict_path))
    q_expanded = expand_query_by_abbr(query_zh, abbr)

    # 2) 稠密檢索（cosine = 內積，向量已 L2）
    qv = retr.encode_query(q_expanded)
    D, I = retr.index.search(qv, K)  # 先取 K 候選
    cand = {idx: float(score) for idx, score in zip(I[0].tolist(), D[0].tolist()) if idx != -1}

    # 2.b（可選）翻譯分支（中文→英文再搜）
    if use_translate:
        # 為了減少依賴，這裡示範極簡規則翻譯（你可改成 Marian/NLLB 模型）
        q_en = q_expanded  # TODO: 如需真翻譯，接入離線翻譯模型
        qv2 = retr.encode_query(q_en)
        D2, I2 = retr.index.search(qv2, K)
        for idx, score in zip(I2[0].tolist(), D2[0].tolist()):
            if idx == -1:
                continue
            cand[idx] = max(cand.get(idx, 0.0), float(score))

    # 3)（可選）BM25 混合
    bm25_scores: Dict[int, float] = {}
    if use_bm25:
        docs = [m.get('text', '') for m in retr.metas]
        if HAS_BM25:
            tokenized = [doc.split() for doc in docs]
            bm25 = BM25Okapi(tokenized)
            q_tok = q_expanded.split()
            sc = bm25.get_scores(q_tok)
            bm25_scores = {i: float(sc[i]) for i in cand.keys()}
            # 正規化到 [0,1]
            if bm25_scores:
                arr = np.array(list(bm25_scores.values()), dtype=np.float32)
                mn, mx = float(arr.min()), float(arr.max())
                rng = max(mx - mn, 1e-6)
                for k in list(bm25_scores.keys()):
                    bm25_scores[k] = (bm25_scores[k] - mn) / rng
        elif HAS_TFIDF:
            vec = TfidfVectorizer(min_df=1)
            X = vec.fit_transform(docs)
            qv_t = vec.transform([q_expanded])
            sims = cosine_similarity(qv_t, X)[0]
            bm25_scores = {i: float(sims[i]) for i in cand.keys()}
        # 混合：alpha*cos + (1-alpha)*bm25
        for i in list(cand.keys()):
            cand[i] = alpha * cand[i] + (1 - alpha) * bm25_scores.get(i, 0.0)

    # 3.1 段落類型加權（參考文獻扣分、表格/流程加分）
    # 先把疑似參考文獻的候選直接移除
    cand = {i: sc for i, sc in cand.items() if not is_bibliography(retr.metas[i])}
    for i in list(cand.keys()):
        cand[i] += type_bonus(retr.metas[i])


    # 4) 輕量重排與去重
    

    # 4.2 同頁連續命中加分（來源集中度）
    page_hits: Dict[int, int] = {}
    for i in cand.keys():
        p = int(retr.metas[i].get('page', -1))
        page_hits[p] = page_hits.get(p, 0) + 1
    for i in list(cand.keys()):
        p = int(retr.metas[i].get('page', -1))
        cand[i] += 0.01 * min(page_hits.get(p, 0), 5)  # 最多 +0.05

    # 4.3 相似句去重：Jaccard 或向量 cos > 閾值
    # 先按分數排序，逐一加入，若與已選任何一條過相似則跳過
    sorted_idx = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)
    selected: List[Tuple[int, float]] = []
    for idx, sc in sorted_idx:
        text = retr.metas[idx].get('text', '')
        keep = True
        for sidx, _ in selected:
            other = retr.metas[sidx].get('text', '')
            # Jaccard 快速去重
            if jaccard_ngrams(text, other, n=3) >= 0.9:
                keep = False
                break
        if not keep:
            continue
        # 若可取得向量（可選）：用 faiss 做 pair 相似比對成本較高，這裡略過
        selected.append((idx, sc))
        if len(selected) >= max(top_n * 3, top_n):  # 保留少量冗餘再截斷
            break

    selected = selected[:top_n]

    # 5) 整理輸出
    hits = []
    for rank, (idx, sc) in enumerate(selected, 1):
        m = retr.metas[idx]
        sc += label_bonus(m, abbr, query_zh)  # 最後再疊一次 label bonus（保小幅影響）
        hit = {
            'rank': rank,
            'score': round(float(sc), 4) if show_scores else None,
            'text': m.get('text'),
            'page': m.get('page'),
            'section': m.get('section'),
            'source': m.get('source'),
            'chunk_id': m.get('chunk_id'),
        }
        if not show_scores:
            hit.pop('score', None)
        hits.append(hit)

    return {'query': query_zh, 'hits': hits}


# ------------- CLI 測試 -------------
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser('retrieve api (no LLM)')
    ap.add_argument('--q', required=True, help='中文查詢')
    ap.add_argument('--topn', type=int, default=5)
    ap.add_argument('--index', default='data/index')
    ap.add_argument('--dict', default='data/dicts/abbr_zh2en.json')
    ap.add_argument('--model', default='intfloat/multilingual-e5-base')
    ap.add_argument('--prefix', choices=['e5','none'], default='e5')
    ap.add_argument('--K', type=int, default=20)
    ap.add_argument('--bm25', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--translate', action='store_true')
    ap.add_argument('--no-repair-meta', dest='repair_meta', action='store_false')
    args = ap.parse_args()

    out = search(args.q, top_n=args.topn, index_dir=args.index, dict_path=args.dict,
                 model=args.model, prefix=args.prefix, K=args.K,
                 use_bm25=args.bm25, alpha=args.alpha, use_translate=args.translate,
                 repair_meta=args.repair_meta)
    print(json.dumps(out, ensure_ascii=False, indent=2))
