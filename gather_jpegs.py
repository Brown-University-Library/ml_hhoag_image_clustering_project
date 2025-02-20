# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#    "httpx",
# ]
# ///


import logging
import pprint
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


ORG_ITEM_URL = 'https://repository.library.brown.edu/api/items/bdr:9r3a8c4a/'


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


def main() -> None:
    pids_and_IDs: list[dict[str, str]] = fetch_parts(ORG_ITEM_URL)
    log.info(f'found ``{len(pids_and_IDs)}`` entries')
    log.info(f'pids_and_IDs: ``{pprint.pformat(pids_and_IDs)}``')


if __name__ == '__main__':
    main()
