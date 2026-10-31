class ValenError(Exception):
    """Base exception for Valen SDK failures."""


class ValenAPIError(ValenError):
    def __init__(self, message, *, status_code=None, request_id=None, debug=None,
                 code=None, model=None, retryable=None, field=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.debug = debug
        self.code = code or ("http_error" if status_code else "request_error")
        self.model = model
        self.retryable = status_code in {429, 502, 503, 504} if retryable is None else retryable
        self.field = field


class ValenResponseError(ValenAPIError):
    """A response violates the decision contract; never retry as a network error."""

    def __init__(self, message, **metadata):
        super().__init__(message, code="invalid_response", retryable=False, **metadata)


class ValenTimeoutError(ValenAPIError):
    """The caller's total request budget expired."""

    def __init__(self, message="Valen total timeout exceeded", **metadata):
        super().__init__(message, code="timeout", retryable=False, **metadata)


class ModelCompatibilityError(ValenError):
    """The requested artifact cannot run with this runtime configuration."""

    code = "model_incompatible"
    retryable = False


class ValenInputError(ValueError, ValenError):
    """A structurally invalid typed request; fix it before retrying."""

    code = "invalid_input"
    retryable = False

    def __init__(self, message, *, field=None):
        super().__init__(message)
        self.field = field
