"""Command-line process entrypoint."""

from __future__ import annotations

import asyncio
import logging

from .config import Settings
from .worker import run_worker


def main() -> None:
    """Run the configured outbound worker until interrupted."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run_worker(Settings()))


if __name__ == "__main__":
    main()
