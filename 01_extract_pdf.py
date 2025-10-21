#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_extract_pdf.py  —  Extract text + (manual) Tesseract OCR + build flow/table nodes

本腳本依照你訂的 01 流程：
1) 逐頁抽取 PDF 文字區塊（不自動判斷是否 OCR）
2) 針對你指定的流程圖頁（--flow-pages），用 **Tesseract** OCR 取得文字行與座標
3) 把 OCR 行群組為「節點句」並輸出 nodes（每框≈一節點；菱形≈條件；箭頭文字嘗試推斷 next_nodes）
4) 針對你指定的表格頁（--table-pages），**不經 OCR**，用 Camelot 解析表格為 DataFrame，
   將每一列轉為一條 RAG 可檢索的節點句並輸出

輸出：
- data/extracted/page_XX.json           # 純文字抽取（blocks）
- data/ocr/page_XX.ocr.json             # OCR 行（text/conf/bbox）
- data/flows/page_XX.nodes.json         # 流程節點（flow nodes）
- data/flows/page_XX.table_nodes.json   # 表格節點（table nodes）

需求：
- pip install pymupdf pytesseract pillow camelot-py[cv] pandas
- Windows 請先安裝 Tesseract 本體，並確認 chi_tra 語言包；程式可用 --tesseract-exe 指定 exe 路徑。

用法：
python src/01_extract_pdf.py  --pdf "data/raw/乳癌治療指引第28版(2024.04.22檢視).pdf"   --flow-pages 3,4,5,6,7,8,9,10   --table-pages 13,14,15,24,25,26,29,30  --ref-pages 22,34,35,37
"""
from __future__ import annotations
import argparse
import io
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# ====== 可選：表格解析（非 OCR） ======
try:
    import camelot  # type: ignore
    import pandas as pd  # type: ignore
    HAS_CAMELOT = True
except Exception:
    HAS_CAMELOT = False

# ---------------------- 基本抽取 ----------------------

def extract_blocks(page: fitz.Page) -> List[Dict]:
    blocks = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        text = (text or "").strip()
        if not text:
            continue
        kind = "paragraph"
        # 簡單 heading 偵測（可自行調整或去除）
        if len(text.split()) <= 12 and re.match(r"^[\w\s/().,\-–%+]+$", text):
            kind = "heading" if (text.isupper() or "/" in text) else "paragraph"
        blocks.append({"type": kind, "bbox": [x0, y0, x1, y1], "text": text})
    return blocks


def page_to_png_bytes(page: fitz.Page, dpi: int = 300) -> bytes:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return pix.tobytes("png")

# ---------------------- Tesseract OCR ----------------------

def run_tesseract_ocr(img_bytes: bytes, lang: str = "eng+chi_tra") -> List[Tuple[str, float, List[int]]]:
    """回傳 [(text, conf, bbox)]，bbox = [x0,y0,x1,y1]（像素）。"""
    img = Image.open(io.BytesIO(img_bytes))
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    lines: List[Tuple[str, float, List[int]]] = []
    n = len(data["text"]) if "text" in data else 0
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        conf_str = data.get("conf", ["-1"] * n)[i]
        try:
            conf = float(conf_str)
        except Exception:
            conf = -1.0
        if conf < 50:  # 0~100；可調
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        lines.append((txt, conf / 100.0, [int(x), int(y), int(x + w), int(y + h)]))
    return lines

# ---------------------- Flow nodes 建立 ----------------------

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = s.replace("：", ":").replace("–", "-").replace("→", " -> ").replace("⇨", " -> ")
    return s


def bbox_merge(a: List[float], b: List[float]) -> List[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def cluster_lines_to_groups(lines: List[Tuple[str, float, List[int]]], y_gap: int = 18, x_tol: int = 80):
    """把 OCR 行以 y0 先排序，依 y 近、x 範圍重疊聚為同一群；粗略對應同一框/段落。"""
    lines = sorted(lines, key=lambda t: (t[2][1], t[2][0]))
    groups: List[Dict] = []

    def x_overlap(b1, b2):
        return min(b1[2], b2[2]) - max(b1[0], b2[0])

    for t, conf, bb in lines:
        placed = False
        for g in groups:
            if bb[1] - g["bbox"][3] <= y_gap and x_overlap(bb, g["bbox"]) >= -x_tol:
                g["lines"].append((t, conf, bb))
                g["bbox"] = bbox_merge(g["bbox"], bb)
                placed = True
                break
        if not placed:
            groups.append({"bbox": bb[:], "lines": [(t, conf, bb)]})
    return groups


def classify_node_text(text: str):
    """粗分：包含 if/criteria/是否 → condition；包含 treat/perform/biopsy/surgery/RT/-> → action。"""
    t = text.lower()
    condition = ""
    action = ""
    if any(k in t for k in ["if ", "criteria", "是否", "若", "when ", "if:"]):
        condition = text
    if any(k in t for k in ["treat", "therapy", "perform", "biopsy", "surgery", "radiotherapy", "give ", "administer", " -> "]):
        action = text
    if not (condition or action):
        (condition if len(text) <= 80 else action)
        if len(text) <= 80:
            condition = text
        else:
            action = text
    return condition, action


def build_flow_nodes(lines: List[Tuple[str, float, List[int]]], page_no: int) -> List[Dict]:
    groups = cluster_lines_to_groups(lines)
    nodes: List[Dict] = []
    for idx, g in enumerate(groups, 1):
        raw = " ".join(normalize_text(t) for t, _, _ in g["lines"])
        raw = re.sub(r"\s*•\s*", " • ", raw)
        cond, act = classify_node_text(raw)
        node = {
            "node_id": f"p{page_no:02d}_n{idx:02d}",
            "condition": cond,
            "action": act,
            "next": [],
            "source": f"P {page_no:02d}/?? flow",
            "bbox": [round(v, 1) for v in g["bbox"]],
        }
        nodes.append(node)

    # 嘗試從箭頭文字推測 next_nodes（best-effort）
    import difflib

    def best_match(token: str):
        scores = []
        for i, n in enumerate(nodes):
            text = (n["condition"] or n["action"]) or ""
            s = difflib.SequenceMatcher(None, token.lower(), text.lower()).ratio()
            scores.append((s, i))
        scores.sort(reverse=True)
        return scores[0][1] if scores and scores[0][0] >= 0.6 else None

    for i, n in enumerate(nodes):
        arrow_parts = re.findall(r"(?:->)\s*([A-Za-z0-9 /()+\-–]+)", (n["condition"] + " " + n["action"]))
        for tok in arrow_parts:
            j = best_match(tok.strip())
            if j is not None and j != i:
                nid = nodes[j]["node_id"]
                if nid not in n["next"]:
                    n["next"].append(nid)
    return nodes

# ---------------------- Table nodes（非 OCR） ----------------------

def df_row_to_text(df, header=None) -> List[str]:
    texts: List[str] = []
    if header is None and len(df) >= 2:
        header = [str(x).strip() for x in df.iloc[0].tolist()]
        body = df.iloc[1:]
    else:
        header = header or [f"C{i+1}" for i in range(df.shape[1])]
        body = df
    for _, row in body.iterrows():
        cells = []
        for c, val in zip(header, row.tolist()):
            v = str(val).strip()
            if v and v.lower() != "nan":
                cells.append(f"{c}={v}")
        text = "; ".join(cells)
        if text:
            texts.append(text)
    return texts


def parse_tables_to_nodes(pdf_path: str, pages: List[int], mode: str = "auto", min_cols: int = 2, min_rows: int = 2):
    if not HAS_CAMELOT:
        print("[warn] camelot 未安裝；跳過表格解析。pip install camelot-py[cv] pandas")
        return {}
    if not pages:
        return {}
    page_spec = ",".join(str(p) for p in pages)

    def try_read(flavor: str):
        try:
            return camelot.read_pdf(pdf_path, pages=page_spec, flavor=flavor)
        except Exception:
            return None

    tables = None
    if mode == "lattice":
        tables = try_read("lattice")
    elif mode == "stream":
        tables = try_read("stream")
    else:
        tables = try_read("lattice") or try_read("stream")

    nodes_by_page: Dict[int, List[Dict]] = {}
    if tables is None:
        return nodes_by_page

    for t in tables:
        p = int(t.page)
        df = t.df
        df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all", axis=0).dropna(how="all", axis=1)
        if df.shape[1] < min_cols or df.shape[0] < min_rows:
            continue
        texts = df_row_to_text(df)
        nodes_by_page.setdefault(p, [])
        base_idx = len(nodes_by_page[p]) + 1
        for k, txt in enumerate(texts, base_idx):
            nodes_by_page[p].append({
                "node_id": f"p{p:02d}_tbl_{k:03d}",
                "condition": "",
                "action": txt,
                "next": [],
                "source": f"P {p:02d}/?? table",
                "bbox": None,
            })
    return nodes_by_page

# ---------------------- 主流程 ----------------------

def parse_pages_arg(pages_str: str | None) -> List[int]:
    if not pages_str:
        return []
    return [int(x) for x in re.split(r"[\s,]+", pages_str.strip()) if x]


def main():
    ap = argparse.ArgumentParser("01_extract_pdf: text + Tesseract OCR (manual pages) + flow/table nodes")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--flow-pages", help="流程圖 OCR 頁碼，如 '4,5,8'")
    ap.add_argument("--table-pages", help="表格解析頁碼（非 OCR），如 '15,16,17'")
    ap.add_argument("--ref-pages", help="參考文獻頁（整頁跳過抽取），如 '22,23-24'")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--tesseract-exe", help="tesseract.exe 的完整路徑（未加到 PATH 時使用）")
    ap.add_argument("--out-extracted", default="data/extracted")
    ap.add_argument("--out-ocr", default="data/ocr")
    ap.add_argument("--out-flows", default="data/flows")
    ap.add_argument("--table-mode", choices=["auto", "lattice", "stream"], default="auto")
    args = ap.parse_args()

    if args.tesseract_exe:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_exe

    pdf_path = Path(args.pdf)
    out_ex = Path(args.out_extracted); out_ex.mkdir(parents=True, exist_ok=True)
    out_ocr = Path(args.out_ocr); out_ocr.mkdir(parents=True, exist_ok=True)
    out_flow = Path(args.out_flows); out_flow.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    flow_pages = set(parse_pages_arg(args.flow_pages))
    table_pages = set(parse_pages_arg(args.table_pages))
    ref_pages = set(parse_pages_arg(args.ref_pages))
    # 參考文獻頁全面跳過：從 flow/table 指定中也排除，避免誤跑
    flow_pages -= ref_pages
    table_pages -= ref_pages


    # 1) 每頁：抽取文字 blocks
    for pno, page in enumerate(doc, start=1):
        if pno in ref_pages:
            # 直接略過此頁（不輸出 extracted/ocr/flows/table）
            (out_ex / f"page_{pno:02d}.json").write_text(
                json.dumps({"page": pno, "skipped": "references"}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"SKIP(refs): page {pno:02d}")
            continue
        blocks = extract_blocks(page)
        ex_payload = {"page": pno, "blocks": blocks}
        (out_ex / f"page_{pno:02d}.json").write_text(
            json.dumps(ex_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 2) 指定流程頁：Tesseract OCR + Flow nodes
    for pno in sorted(flow_pages):
        page = doc[pno - 1]
        img_bytes = page_to_png_bytes(page, dpi=args.dpi)
        lines = run_tesseract_ocr(img_bytes, lang="eng+chi_tra")  # [(text, conf, bbox)]
        ocr_payload = {
            "page": pno,
            "engine": "tesseract",
            "lines": [{"text": t, "conf": float(c), "bbox": b} for t, c, b in lines],
        }
        (out_ocr / f"page_{pno:02d}.ocr.json").write_text(
            json.dumps(ocr_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        nodes = build_flow_nodes(lines, pno)
        (out_flow / f"page_{pno:02d}.nodes.json").write_text(
            json.dumps({"page": pno, "nodes": nodes}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✓ page {pno:02d}: OCR lines={len(lines)} → flow nodes={len(nodes)}")

    # 3) 指定表格頁：Camelot 解析 → Table nodes（非 OCR）
    if table_pages:
        nodes_by_page = parse_tables_to_nodes(str(pdf_path), sorted(table_pages), mode=args.table_mode)
        for p, nodes in nodes_by_page.items():
            outp = out_flow / f"page_{p:02d}.table_nodes.json"
            outp.write_text(json.dumps({"page": p, "nodes": nodes}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✓ page {p:02d}: table nodes={len(nodes)} → {outp}")
    
    # 紀錄頁面旗標
    meta_dir = Path("data/meta"); meta_dir.mkdir(parents=True, exist_ok=True)
    page_flags = {
        "refs": sorted(ref_pages),
        "ocr": sorted(flow_pages),     # 實際 OCR 過的流程頁（已扣除 ref）
        "tables": sorted(table_pages)  # 實際表格解析過的頁（已扣除 ref）
    }
    (meta_dir / "page_flags.json").write_text(
        json.dumps(page_flags, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("完成：extracted/  ocr/  flows/")


if __name__ == "__main__":
    main()
