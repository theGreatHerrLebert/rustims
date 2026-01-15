"""Database search tools for timsTOF DDA data."""

# Note: search_dda imports are deferred to avoid circular imports
# Use: from imspy.timstof.dbsearch.search_dda import SearchConfig, run_search_pipeline

__all__ = [
    "SearchConfig",
    "run_search_pipeline",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in ("SearchConfig", "run_search_pipeline"):
        from .search_dda import SearchConfig, run_search_pipeline
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
