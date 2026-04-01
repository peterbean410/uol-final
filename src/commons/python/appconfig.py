"""
Global application configuration loaded from environment variables.

Usage:
    from commons.python.appconfig import AppConfig

    config = AppConfig()
    print(config.app_client_id)
    print(config.s3_bucket)
"""

import os
import time
from pathlib import Path

from ctrader_open_api import Auth
from dotenv import load_dotenv

# Refresh the token if it expires within this many seconds
_TOKEN_EXPIRY_BUFFER_SECONDS = 60


class AppConfig:
    """Holds values for environment variables used across all modules.

    Loads from a .env file (defaults to the project-level markets/.env)
    and exposes each variable as an attribute.
    """

    _DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

    def __init__(self, env_path: str | Path | None = None):
        env_path = Path(env_path) if env_path else self._DEFAULT_ENV_PATH
        load_dotenv(dotenv_path=str(env_path))

        self.app_client_id: str = os.getenv("APP_CLIENT_ID", "")
        self.app_client_secret: str = os.getenv("APP_CLIENT_SECRET", "")
        self.access_token: str = os.getenv("ACCESS_TOKEN", "")
        self.refresh_token: str = os.getenv("REFRESH_TOKEN", "")
        self.redirect_uri: str = os.getenv("REDIRECT_URI", "http://localhost:8765/callback")
        self.default_account_id: int | None = self._int_or_none("DEFAULT_ACCOUNT_ID")
        self.live_account_id: int | None = self._int_or_none("LIVE_ACCOUNT_ID")
        self.default_mode: str = os.getenv("DEFAULT_MODE", "demo")
        self.s3_bucket: str = os.getenv("S3_BUCKET", "")
        self.token_expiry: float = float(os.getenv("TOKEN_EXPIRY", "0"))

        self._env_path = env_path

    @staticmethod
    def _int_or_none(key: str) -> int | None:
        val = os.getenv(key)
        if val:
            try:
                return int(val)
            except ValueError:
                return None
        return None

    def refresh_access_token(self) -> None:
        """Refresh the access token if it is expired or soon-to-expire.

        Checks TOKEN_EXPIRY against the current time (with a 60-second buffer).
        If the token is still valid, this method is a no-op.

        Raises:
            RuntimeError: If no refresh token is available or the refresh call fails.
        """
        now = time.time()
        if self.access_token and self.token_expiry > now + _TOKEN_EXPIRY_BUFFER_SECONDS:
            return

        if not self.refresh_token:
            raise RuntimeError(
                "Access token expired and no REFRESH_TOKEN available. "
                "Run get_tokens.py to re-authenticate."
            )

        auth = Auth(self.app_client_id, self.app_client_secret, self.redirect_uri)
        token = auth.refreshToken(self.refresh_token)

        new_access = token.get("accessToken")
        new_refresh = token.get("refreshToken", self.refresh_token)
        expires_in = int(token.get("expiresIn", 0))

        if not new_access:
            raise RuntimeError("Token refresh failed; no accessToken returned.")

        self.access_token = new_access
        self.refresh_token = new_refresh
        self.token_expiry = time.time() + expires_in

        self._upsert_env_vars({
            "ACCESS_TOKEN": new_access,
            "REFRESH_TOKEN": new_refresh,
            "TOKEN_EXPIRY": str(int(self.token_expiry)),
        })

    def update_tokens(self, access_token: str, refresh_token: str) -> None:
        """Update access and refresh tokens in memory and persist to .env."""
        self.access_token = access_token
        self.refresh_token = refresh_token
        self._upsert_env_vars({"ACCESS_TOKEN": access_token, "REFRESH_TOKEN": refresh_token})

    def _upsert_env_vars(self, updates: dict[str, str]) -> None:
        """Update key=value pairs in the .env file."""
        if not self._env_path.exists():
            self._env_path.write_text("", encoding="utf-8")

        lines = self._env_path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
        output_lines = []
        seen = set()

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                output_lines.append(raw)
                continue
            k = line.split("=", 1)[0].strip()
            if k in updates:
                output_lines.append(f"{k}={updates[k]}")
                seen.add(k)
            else:
                output_lines.append(raw)

        for k in set(updates.keys()) - seen:
            output_lines.append(f"{k}={updates[k]}")

        self._env_path.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")

    @property
    def account_id(self) -> int | None:
        """Return live account ID if set, otherwise default account ID."""
        return self.live_account_id or self.default_account_id
