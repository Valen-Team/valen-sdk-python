class ValenError(Exception):
    """Base exception for Valen SDK failures."""


class ValenAPIError(ValenError):
    def __init__(self, message, *, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class ModelCompatibilityError(ValenError):
    """The requested artifact cannot run with this runtime configuration."""
