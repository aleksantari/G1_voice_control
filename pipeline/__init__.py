"""Language control pipeline — wires STT, parsers, and validation together."""

__all__ = ["LanguageControlPipeline"]


def __getattr__(name):
    if name == "LanguageControlPipeline":
        from pipeline.pipeline import LanguageControlPipeline
        return LanguageControlPipeline
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
