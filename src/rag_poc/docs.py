"""検証に使う資料の定義と、ページごとのテキスト抽出。

ページ番号は PDF の物理ページ(1 始まり)で数える。資料に印刷されたページ番号とは一致しない。
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    filename: str
    # 左右の余白にある章見出し(縦書きの見出し)の幅。この範囲に収まるテキストは本文から除く
    side_margin: float = 0

    @property
    def path(self) -> Path:
        return DATA_DIR / self.filename


DOCS = [
    Doc("main", "中小企業の情報セキュリティ対策ガイドライン第4.0版 本編", "sme_guideline_v4.0.pdf", side_margin=42),
    Doc("app5", "付録5 情報セキュリティ関連規程(サンプル)", "sme_guideline_v4.0_app_5.pdf"),
    Doc("app3", "付録3 5分でできる!情報セキュリティ自社診断", "sme_guideline_v4.0_app_3.pdf"),
]
DOCS_BY_ID = {d.id: d for d in DOCS}


def extract_pages(doc: Doc) -> list[str]:
    """資料の各ページのテキストを、物理ページ順に返す。"""
    with pymupdf.open(doc.path) as pdf:
        return [page.get_text() for page in pdf]


def clean_pages(doc: Doc) -> list[str]:
    """チャンク分割と索引作成に使う、余白の章見出しを除いたページごとのテキストを返す。"""
    if not doc.side_margin:
        return extract_pages(doc)
    pages = []
    with pymupdf.open(doc.path) as pdf:
        for page in pdf:
            left, right = doc.side_margin, page.rect.width - doc.side_margin
            blocks = [b for b in page.get_text("blocks") if left < b[2] and b[0] < right]
            pages.append("".join(b[4] for b in blocks))
    return pages


def dump_pages(out_dir: Path = DATA_DIR / "text") -> None:
    """全資料のページごとのテキストを data/text/<doc>/p001.txt に書き出す。"""
    for doc in DOCS:
        d = out_dir / doc.id
        d.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(extract_pages(doc), start=1):
            (d / f"p{i:03d}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    dump_pages()
