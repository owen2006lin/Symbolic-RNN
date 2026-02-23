from __future__ import annotations
import arxiv
import pandas as pd
from tqdm import tqdm
import re
import json
import csv
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import gzip
import argparse

#Use re.compile() to efficiently reuse regex pattern (otherwise python creates a new one each time)

#Newer arxiv ids are in the format of YYMM.numbers(version optional)eg 2105.12345
NEWSTYLE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
#Old style ids are in format of category(.optional subcategory)/numbers(version optional) eg: cs.AI/0102030
OLDSTYLE = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?\/\d{7}(v\d+)?$", re.IGNORECASE)


def normalize_arxiv_id(aid : str) -> str:
    if not isinstance(aid, str):
        return ""
    #Strip any erroneous whitespace, and also returns empty string in case nothing given
    aid = (aid or "").strip()
    #Substitutes the optional version ending with empty string
    aid = re.sub(r"v\d+$", "", aid)
    return aid

def is_valid_arxiv_id(aid : str) -> bool:
    #Arxiv id must be either new or old style
    return bool(NEWSTYLE.match(aid) or OLDSTYLE.match(aid))


#Class to hold a single CSV row in a structured way, arXiv ID links to arXiv source download to extract equations and for category lookup
@dataclass
class WorkRow:
    openalex_id: str
    doi: str
    title: str
    publication_year: int
    cited_by_count: int
    arxiv_id: Optional[str]
    work_type: str

#Read an OpenAlex CSV file with formatting of build_top10k_OpenAlex():
#    openalex_url, doi_url, title, year, cited_by_count, arxiv_id, type

#Returns WorkRow list. Skips a header row if present.

def read_openalex_csv_file(csv_path: Path) -> List[WorkRow]:

    rows: List[WorkRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        #skip the first line which is just formatting data for the csv: openalex_id,doi,title,publication_year,cited_by_count,arxiv_id,type

        next(reader, None)
        for parts in reader:
            if not parts:
                continue

            # Ensures parts has exactly 7 columns so append more in case something is missing
            while len(parts) < 7:
                parts.append("")

            openalex_id, doi, title, year, cited_by, arxiv_id, work_type = parts[:7]


            year_i = int(year)
            cited_i = int(cited_by)


            arxiv_id = arxiv_id.strip() or None
            #after extracting the data from the csv, create a WorkRow object to append to the output list
            rows.append(
                WorkRow(
                    openalex_id=openalex_id.strip(),
                    doi=doi.strip(),
                    title=title.strip(),
                    publication_year=year_i,
                    cited_by_count=cited_i,
                    arxiv_id=arxiv_id,
                    work_type=work_type.strip(),
                )
            )

    return rows



#Fetch arXiv metadata for a given id using arxiv.Search.

#Returns:
#    (status, primary_category)
#    status:
#    - "ok" if found
#    - "not_found" if no result returned
#    - "metadata_error" on exceptions

def fetch_arxiv_metadata(arxiv_id: str) -> Tuple[str, Optional[str]]:

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        #using next to access the first element of query (there is only one element)
        result = next(search.results(), None)
        if result is None:
            return ("not_found", None)
        #getattr extracts the category
        primary_category = getattr(result, "primary_category", None)
        return ("ok", primary_category)
    except Exception:
        return ("metadata_error", None)
    

#Download the arXiv source package and extract it into out_dir.
#
#Returns status:
#    - "ok" if extracted
#    - "download_error" if download failed
#    - "extract_error" if tar extraction failed
#
def download_and_extract_arxiv_source(arxiv_id: str, out_dir: Path) -> str:

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(search.results(), None)
        if result is None:
            return "download_error"
        #downloading  arXiv query to local (will delete later)
        tar_path = Path(result.download_source(dirpath=str(out_dir)))
    except Exception:
        return "download_error"
    try:
        raw_head = tar_path.read_bytes()[:200].lstrip()
        if raw_head.startswith(b"%PDF"):
            return "pdf_source"
        if raw_head.startswith(b"%!PS-Adobe"):
            return "ps_source"
    except Exception:
        pass
    try:
        try:
            #attempts to extract tar files (this currently does not work for all files, will need fixing in the future)
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(path=out_dir)
            return "ok"
        except Exception:
            pass
        try:
            out_dir.mkdir(parents = True, exist_ok = True)
            with gzip.open(tar_path, "rb") as f:    
                tex_bytes = f.read()
            (out_dir/"main.tex").write_bytes(tex_bytes)
            return "ok"
        except Exception:
            return "not tex nor directory"
        
    except Exception:
        return "extract_error"

def is_probably_binary_file(path: Path, sample_bytes: int = 4096) -> bool:
    """
    Heuristic to detect binary files:
      - contains NUL byte
      - too many control characters
    """
    try:
        data = path.read_bytes()[:sample_bytes]
    except Exception:
        return True
    if b"\x00" in data:
        return True
    nontext = sum(1 for b in data if b < 9 or (b > 13 and b < 32))
    return nontext > 0.30 * max(1, len(data))

#Recursively scan root_dir and return:
#    - tex_files: list of .tex files
#    - skipped_binary: count of binary-like files
#    - skipped_notlatex: count of non-binary but non-.tex files


def collect_tex_files(root_dir: Path) -> Tuple[List[Path], int, int]:

    tex_files: List[Path] = []
    skipped_binary = 0
    skipped_notlatex = 0

    # common binary extensions (fast skip)
    binary_exts = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".eps", ".zip", ".gz", ".tar"}

    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() in binary_exts:
            skipped_binary += 1
            continue

        if is_probably_binary_file(p):
            skipped_binary += 1
            continue

        if p.suffix.lower() == ".tex":
            tex_files.append(p)
        else:
            skipped_notlatex += 1

    return tex_files, skipped_binary, skipped_notlatex

EQUATION_ENVS = [
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*", "flalign", "flalign*",
    "alignat", "alignat*"
]

#(" + " | ".join(...) + ") -> Group 1 captures 
#   \begin{e}
#   .
#   .
#   .
#   \end{e}
#   For any environment in the list above


#(.*?) -> Group 2 captures anything inide the Group1 environment

ENV_PATTERN = re.compile(
    r"\\begin\{(" + "|".join(re.escape(e) for e in EQUATION_ENVS) + r")\}(.*?)\\end\{\1\}",
    re.DOTALL
)
#Find sections matching \[...\]
BRACKET_DISPLAY_PATTERN = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
#Find sections matching $$ ... $$
DOLLAR_DISPLAY_PATTERN = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

#Remove LaTeX comments (comments start with %, but preserves lines with \%)
def strip_latex_comments(tex: str) -> str:
    #Regex matches %, (?<!\\) means only if not preceeded by backslash, .* gets rid of everything after
    #So removes entire line if it begins with %, but only if % is not preceeded by \ which could be a real equation
    return re.sub(r"(?<!\\)%.*", "", tex)


#Extract equation blocks from a TeX document.

#Returns list of (kind, latex_block), where kind is one of:
    #- environment:<name>
    #- display:\\[...\\]
    #- display:$$...$$
#Inline $...$ is NOT extracted by default (too noisy).

def extract_equations_from_tex(tex: str) -> List[Tuple[str, str]]:
    #first strip comments
    tex = strip_latex_comments(tex)
    found: List[Tuple[str, str]] = []
    #search for \begin{...} ... \end{...} pattern
    for m in ENV_PATTERN.finditer(tex):
        #environment name
        env = m.group(1)
        #Adds (environment, block) to list
        found.append((f"environment:{env}", m.group(0).strip()))
        
    #searches for other common equation blocks
    for m in BRACKET_DISPLAY_PATTERN.finditer(tex):
        found.append(("display:\\[...\\]", m.group(0).strip()))

    for m in DOLLAR_DISPLAY_PATTERN.finditer(tex):
        found.append(("display:$$...$$", m.group(0).strip()))

    return found

"""
For one WorkRow (must have arxiv_id):
    - fetch metadata (category)
    - download+extract source
    - collect .tex files
    - extract equations
    - return:
        summary_record, equation_records

IMPORTANT:
    Every returned record includes cited_by_count and arxiv_primary_category (may be None).
"""

#returns tuple of (base_summary, equation_records)
def process_work_row(row: WorkRow, keep_equations: bool,) -> Tuple[dict, List[dict]]:

    arxiv_id = normalize_arxiv_id(row.arxiv_id or "")

    # Ensure required fields always present in output
    base_summary = {
        "arxiv_id": arxiv_id if arxiv_id else None,
        "cited_by_count": row.cited_by_count,
        "arxiv_primary_category": None,  # will fill if possible
        "status": "init",
        "tex_files": 0,
        "skipped_binary": 0,
        "skipped_notlatex": 0,
        "equations": 0,
    }

    equation_records: List[dict] = []

    if not arxiv_id or not is_valid_arxiv_id(arxiv_id):
        base_summary["status"] = "not_arxiv"
        return base_summary, equation_records

    # Metadata (category)
    meta_status, primary_cat = fetch_arxiv_metadata(arxiv_id)
    base_summary["arxiv_primary_category"] = primary_cat

    if meta_status != "ok":
        # We *still* try to download sources; category may remain None.
        # But status should reflect metadata problem only if download succeeds.
        pass

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        #downloads and extracts arXiv ip from API to temp directory
        dl_status = download_and_extract_arxiv_source(arxiv_id, tmp_dir)
        if dl_status != "ok":
            base_summary["status"] = dl_status if meta_status == "ok" else f"{meta_status}+{dl_status}"
            return base_summary, equation_records

        #Find all latex files from directory after zip extracted
        tex_files, skipped_binary, skipped_notlatex = collect_tex_files(tmp_dir)

        eq_count = 0
        for tex_path in tex_files:
            try:
                content = tex_path.read_text(errors="ignore")
            except Exception:
                continue
            #extract all equation-like blocks and add to list
            extracted = extract_equations_from_tex(content)
            for idx, (kind, latex_block) in enumerate(extracted):
                eq_count += 1
                if keep_equations:
                    equation_records.append({
                        "arxiv_id": arxiv_id,
                        "cited_by_count": row.cited_by_count,
                        "arxiv_primary_category": primary_cat,
                        "file": str(tex_path.relative_to(tmp_dir)),
                        "kind": kind,
                        "index": idx,
                        "latex": latex_block,
                    })
        #updates summary for this paper
        base_summary.update({
            "status": "ok" if meta_status == "ok" else f"{meta_status}+ok",
            "tex_files": len(tex_files),
            "skipped_binary": skipped_binary,
            "skipped_notlatex": skipped_notlatex,
            "equations": eq_count,
        })

        return base_summary, equation_records
    
    #Full workflow, takes in csv, and creates json of all extracted equations and summary for the entire paper
def build_equation_dataset_from_openalex_csv_file(csv_path: Path,
    out_summary_jsonl: Path,
    out_equations_jsonl: Optional[Path] = None,
    max_papers: Optional[int] = None,
) -> None:
    """
    Full pipeline:
      - read OpenAlex CSV file
      - filter rows with arxiv_id
      - process each row
      - write JSONL:
          summary.jsonl (required)
          equations.jsonl (optional)
    """
    #opens csv
    rows = read_openalex_csv_file(csv_path)
    rows = [r for r in rows if r.arxiv_id]

    if max_papers is not None:
        rows = rows[:max_papers]


    out_summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_equations_jsonl is not None:
        out_equations_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_summary_jsonl.open("w", encoding="utf-8") as fsum:
        feq = out_equations_jsonl.open("w", encoding="utf-8") if out_equations_jsonl else None
        try:
            for row in rows:
                summary, eqs = process_work_row(row, keep_equations=(feq is not None))
                # Guarantee required fields
                assert "cited_by_count" in summary
                assert "arxiv_primary_category" in summary
                #update summary.jsonl with metadata
                fsum.write(json.dumps(summary, ensure_ascii=False) + "\n")
                #writes in every equation + metadata as one line
                if feq is not None:
                    for rec in eqs:
                        # Also guarantee required fields per equation record
                        assert "cited_by_count" in rec
                        assert "arxiv_primary_category" in rec
                        feq.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            if feq is not None:
                feq.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build equation dataset from OpenAlex CSV"
    )

    parser.add_argument(
        "max_papers",
        type=int,
        nargs="?",
        default=None,
        help="Maximum number of papers to process (omit for full run)"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="Test_openalex_raw.csv",
        help="Path to OpenAlex CSV file"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="out",
        help="Output directory"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_equation_dataset_from_openalex_csv_file(
        csv_path=csv_path,
        out_summary_jsonl=out_dir / "summary.jsonl",
        out_equations_jsonl=out_dir / "equations.jsonl",
        max_papers=args.max_papers,
    )

    print("Done:", out_dir / "summary.jsonl", "and", out_dir / "equations.jsonl")
