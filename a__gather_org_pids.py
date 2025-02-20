# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#    "httpx",
# ]
# ///

"""
Gathers HH018977 / AFL-CIO org pids.

Usage:
$ uv run ./a__gather_org_pids.py
"""

import json
import logging
import pprint
import sys
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


PROJECT_DIR_NAME = 'ml_hhoag_image_clustering_project'
ORG_ITEM_URL = 'https://repository.library.brown.edu/api/items/bdr:9r3a8c4a/'
ORG_PIDS_JSON_PATH = '../output_data/a__afl_cio_HH018977_org_pids.json'


def check_cwd() -> None:
    cwd = Path.cwd()
    log.debug(f'cwd: ``{cwd}``')
    if not cwd.name == PROJECT_DIR_NAME:
        print(f"ERROR: cd into the project directory; you're at:: ``{cwd}``")
        sys.exit(1)
    return


def fetch_parts(url: str) -> list[dict[str, str]]:
    """
    Returns a list of dictionaries with keys 'pid' and 'mods_id_bdr_pid_ssim', like:
    [ {'pid': 'bdr:gbzad5mb', 'mods_id_bdr_pid_ssim': 'HH018977_0001'}, {etc...} }
    """
    with httpx.Client() as client:
        response: httpx.Response = client.get(url)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    ## get the relations-->hasPart data -----------------------------
    has_parts: list[dict[str, Any]] = data.get('relations', {}).get('hasPart', [])
    pids_and_IDs: list[dict[str, str]] = []
    for part in has_parts:
        ## build the pid and mods_id list ---------------------------
        part: dict[str, Any] = part
        pid: str | None = part.get('pid')
        mods_id_list: list[str] = part.get('mods_id_bdr_pid_ssim', [])
        mods_id: str | None = mods_id_list[0] if mods_id_list else None
        if pid is not None and mods_id is not None:
            pids_and_IDs.append({'pid': pid, 'mods_id_bdr_pid_ssim': mods_id})
    return pids_and_IDs


def export_json(pids_and_IDs: list) -> None:
    save_path = Path(ORG_PIDS_JSON_PATH).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)  # create parent dirs if needed
    log.debug(f'save_path: ``{save_path}``')
    with open(save_path, 'w') as f:
        json.dump(pids_and_IDs, f, sort_keys=True, indent=2)
    return


def main() -> None:
    check_cwd()
    pids_and_IDs: list[dict[str, str]] = fetch_parts(ORG_ITEM_URL)
    log.info(f'found ``{len(pids_and_IDs)}`` entries')
    log.info(f'pids_and_IDs: ``{pprint.pformat(pids_and_IDs)}``')
    export_json(pids_and_IDs)
    return


if __name__ == '__main__':
    main()
