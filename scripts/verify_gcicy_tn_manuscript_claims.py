#!/usr/bin/env python3
"""Compatibility entry point for the current final-table verifier.

An earlier version maintained a brittle ledger of exact phrases from an older
manuscript revision.  That ledger has been retired.  The authoritative check
now regenerates all five final tables from their frozen inputs, verifies the
input hashes, parses every row and column, checks manuscript input order, and
enforces the current wording gates.
"""

from verify_gcicy_tn_final_tables import main


if __name__ == "__main__":
    main()
