"""Console-script entrypoint: run the agent as an ACP server over stdio."""

from __future__ import annotations

import asyncio
import logging
import os

from acp import run_agent

from .acp_agent import OfflineAgent


async def _main() -> None:
    await run_agent(OfflineAgent())


def cli() -> None:
    # Logs go to stderr so they don't corrupt the stdio JSON-RPC channel on stdout.
    logging.basicConfig(
        level=os.environ.get("OFFLINE_AGENT_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_main())


if __name__ == "__main__":
    cli()
