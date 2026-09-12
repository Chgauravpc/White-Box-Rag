"""
manual_smoke_bp3.py — manual smoke test for the BRD parser + requirement mapper.

Formerly `backend/test_bp3.py`. It was never a pytest test (no assertions,
runs `asyncio.run(main())` unconditionally at import time, needs a live LLM
API key and an ingested corpus) — pytest's `testpaths = tests` in
backend/pytest.ini correctly never collected it, but its name made that look
like an accidental exclusion rather than an intentional one. Moved here,
under a real `__main__` guard, so it's unambiguously a manual operator tool.

Run with:
    cd backend && python scripts/manual_smoke_bp3.py
"""

import asyncio
import logging

from compliance.brd_parser import parse_brd
from compliance.mapper import map_requirement

logging.basicConfig(level=logging.INFO)


async def main():
    print('=======================================')
    print('1. Testing BRD Parser...')
    print('=======================================')
    reqs = await parse_brd('compliance/sample_brd.txt')
    print(f'Successfully parsed {len(reqs)} requirements.')
    for r in reqs:
        print(f' - [ID: {r.id}] {r.category} | {r.text[:60]}...')

    print('\n=======================================')
    print('2. Testing Requirement Mapper (Retrieval + LLM)...')
    print('=======================================')
    if reqs:
        test_req = reqs[0]
        print(f'Mapping req: {test_req.text}')
        result = await map_requirement(test_req)
        print(f'\nDone! Alignment Score: {result.get("alignment_score")}')
        print(f'Risk Level: {result.get("risk_level")}')
        print(f'Gaps identified: {result.get("gaps")}')
        print(f'Violations: {result.get("violations")}')


if __name__ == "__main__":
    asyncio.run(main())
