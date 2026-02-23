from __future__ import annotations
import arxiv
import pandas as pd
from tqdm import tqdm
import time
import requests
import re
import json
import sys
from dataclasses import dataclass
from pathlib import Path

OPENALEX = "https://api.openalex.org"
arxiv_source_id = "https://openalex.org/S4306400194"

PHYSICS_PREFIXES = (
    "physics.", "astro-ph", "cond-mat", "hep-", "nucl-", "gr-qc", "quant-ph", "math-ph", "nlin"
)
BIOLOGY_PREFIX = "q-bio"

def is_physics(cat : str) -> bool:
    return bool(cat) and cat.startswith(PHYSICS_PREFIXES)

def is_biology(cat : str) -> bool:
    return bool(cat) and cat.startwith(BIOLOGY_PREFIX)

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

#Function inputs url, and returns raw text parsed as json
def get_json(url, params = None, retries = 6, backoff = 1.6):
    for attempt in range(retries):
        #using requests library to pull website data from url
        r = requests.get(url, params = params, timeout = 45)
        #status code 200 on successful return
        if r.status_code == 200:
            return r.json()
        #failure codes, wait before trying again
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff**attempt)
            continue
    
#All Arxiv abstract urls are in the format of arxiv.org/abs/alphanumericid. It's the same thing accordingly for pdf pages
_ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/([^?#/]+)", re.IGNORECASE)
_ARXIV_PDF_RE = re.compile(r"arxiv\.org/pdf/([^?#/]+)", re.IGNORECASE)

#Taking the json from the previous function and extracting the proper arxiv id
def extract_arxiv_id_from_work(work):
    ids = work.get("ids") or {}
    #The path we care about in the OpenAlex hierarchy goes work->locations->pdf_url/landing_page_url->arxiv link
    for loc in (work.get("locations") or []):
        for key in ("landing_page_url", "pdf_url"):
            u = loc.get(key)
            if not u:
                continue
            #the search function in the re. package checks the entire string if it finds a match of the regex pattern
            m = _ARXIV_ABS_RE.search(u) or _ARXIV_PDF_RE.search(u)
            if m:
                #group takes the first () in the regex which will be the id (also removing any possible 'pdf')
                aid = normalize_arxiv_id(m.group(1).replace(".pdf", ""))
                if is_valid_arxiv_id(aid):
                    return aid
    #In many cases, there will exist papers on OpenAlex (eg: Random Forests) but are not on arXiv. Usually, these papers were
    #published in the pre internet days, so no corresponding upload to arXiv were made. For these papers, we'll just return nothing
    #and treat them as if they don't exist (since we can't parse them from arXiv)
    return None

def fetch_openalex_arxiv_works_cursor(max_works, mailto=None):
    #filter for selecting only sources on OpenAlex that are from arXiv
    the_filter = f"locations.source.id:{arxiv_source_id}"
    #selecting only relevant metadata
    select = ",".join([
        "id", "doi", "title", "publication_year", "cited_by_count",
        "ids", "locations", "type"
    ])

    per_page = 200
    cursor = "*"
    #storing final output here in this format -> [{OpenAlex id: _, doi: _, title: _, year: _, citations: _, arXiv id: _, type: _} ...]
    rows = []
    #for visuals, how far until completion
    pbar = tqdm(total=max_works, desc="OpenAlex fetch")

    while len(rows) < max_works:
        params = {
            "filter": the_filter,
            "sort": "cited_by_count:desc",
            "per-page": per_page,
            "cursor": cursor,
            "select": select
        }
        #using get_json as defined earlier, starting from the OpenAlex API and searching through all the top works
        data = get_json(f"{OPENALEX}/works", params=params)
        results = data.get("results", [])
        #end early if miss
        if not results:
            break
        #For each found article:
        for w in results:
            aid = extract_arxiv_id_from_work(w)
            if not aid:
                continue
            #Saving article metadata into rows
            rows.append({
                "openalex_id": w.get("id"),
                "doi": w.get("doi") or "",
                "title": w.get("title") or "",
                "publication_year": w.get("publication_year"),
                "cited_by_count": int(w.get("cited_by_count") or 0),
                "arxiv_id": aid,
                "type": w.get("type") or ""
            })

            if len(rows) >= max_works:
                break

        pbar.update(min(len(results), max_works - pbar.n))
        cursor = data.get("meta", {}).get("next_cursor")

        if not cursor:
            break
        #make sure to wait to not get timed out by the API
        time.sleep(0.1)

    pbar.close()
    return rows


#given the list of arxiv ids, find the categories of these ids
def arxiv_query_categories(arxiv_ids, batch_size = 50):
    #output will be a map: ids->category
    out = {}
    i = 0
    pbar = tqdm(total=len(arxiv_ids), desc = "arXiv category resolution")
    
    while i < len (arxiv_ids):
        #To avoid bottlenecking the arXiv api, only search in chunks of 50
        chunk = arxiv_ids[i:i+batch_size]
        #arXiv.Search returns a search object that contains all the ids we want arXiv to query
        search = arxiv.Search(id_list = chunk)
        #No actual api call is made until .results() is called, results contains all the metadata about the corresponding article
        #we want its id, and map to its primary category
        for result in search.results():
            aid = result.get_short_id()
            out[aid] = result.primary_category
        
        i+=len(chunk)
        pbar.update(len(chunk))
        #wait to avoid bottlenecking
        time.sleep(0.2)
    pbar.close()
    return out

def sort_key(w):
    year = w["publication_year"] if w["publication_year"] else 9999
    return (-w["cited_by_count"], year, w["arxiv_id"])

def save(name, rows, out_prefix):
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_prefix}_{name}.csv", index=False)
    with open(f"{out_prefix}_{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def build_top10k_OpenAlex(oversample_factor = 10, out_prefix = "Test"):
    target = 1000
    fetch_n = target * oversample_factor
    works = fetch_openalex_arxiv_works_cursor(fetch_n)
    pd.DataFrame(works).to_csv(f"{out_prefix}_openalex_raw.csv", index = False)



def main():
    if len(sys.argv) != 2:
        print("Usage: python build_articles_list.py <number>")
        sys.exit(1)
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Argument must be integer.")
        sys.exit(1)
    build_top10k_OpenAlex(n)

if __name__ == "__main__":
    main()

