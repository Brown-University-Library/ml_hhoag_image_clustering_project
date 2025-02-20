# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#    "httpx",
# ]
# ///


import logging
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


ORG_ITEM_URL = 'https://repository.library.brown.edu/api/items/bdr:9r3a8c4a/'


def fetch_pids(url: str) -> list[str]:
    with httpx.Client() as client:
        response: httpx.Response = client.get(url)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    has_parts: list[dict[str, Any]] = data.get('relations', {}).get('hasPart', [])
    pids: list[str] = [part.get('pid') for part in has_parts if part.get('pid') is not None]
    return pids


def main() -> None:
    pids: list[str] = fetch_pids(ORG_ITEM_URL)
    log.info(f'found ``{len(pids)}`` pids')
    log.info(f'pids: ``{pids}``')


if __name__ == '__main__':
    main()
