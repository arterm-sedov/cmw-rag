from typing import Any

from pydantic import BaseModel, field_validator


class HTTPResponse(BaseModel):
    """Validates HTTP response structure to prevent NoneType errors."""

    success: bool
    status_code: int
    raw_response: dict | str | None = None
    error: str | None = None
    base_url: str

    @field_validator("status_code")
    @classmethod
    def validate_status_code(cls, v: int) -> int:
        if not (100 <= v < 600):
            msg = f"Invalid HTTP status code: {v}"
            raise ValueError(msg)
        return v

    @field_validator("raw_response")
    @classmethod
    def validate_raw_response(cls, v: dict | str | None) -> dict | str | None:
        if v is not None and not isinstance(v, (dict, str)):
            msg = f"Invalid raw_response type: {type(v)}"
            raise ValueError(msg)
        return v


class APIResponse(BaseModel):
    """Validates Comindware Platform API response format."""

    response: Any | None = None
    success: bool | None = None
    error: str | None = None


class RequestConfig(BaseModel):
    """Validates server configuration for CMW Platform API."""

    base_url: str
    login: str
    password: str
    timeout: int = 30

    @field_validator("base_url", mode="before")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        v = v.rstrip("/")
        if not v:
            msg = "base_url cannot be empty"
            raise ValueError(msg)
        return v

    @field_validator("login", "password", mode="before")
    @classmethod
    def validate_credentials(cls, v: str) -> str:
        if not v:
            raise ValueError("Credential cannot be empty")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v <= 0:
            msg = "timeout must be positive"
            raise ValueError(msg)
        return v
