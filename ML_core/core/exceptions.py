class MLCoreError(Exception):
    """Base error for ML core failures."""


class ModelArtifactError(MLCoreError):
    """Raised when a model artifact cannot be loaded or used."""


class InvalidIntentError(MLCoreError):
    """Raised when user intent cannot be converted into a usable request."""
