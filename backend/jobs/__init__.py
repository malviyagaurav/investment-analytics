"""Scheduled / batch jobs for the analytics engine.

Currently:
- watchlist: re-rank a curated set of categories on a schedule and
  persist daily snapshots so score / rank trends can be tracked over
  time without the user clicking anything.
"""
