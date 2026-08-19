"""Authenticated, least-privilege LAN gateway for OpenArmSim episode control."""

from __future__ import annotations

import argparse
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import ProxyHandler
from urllib.request import Request
from urllib.request import build_opener

DEFAULT_TOKEN_FILE = Path(".openarm_sim_gateway_token")
DEFAULT_ALLOWED_CIDRS = (
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
)
MAX_BODY_BYTES = 1024
MIN_TOKEN_LENGTH = 32


def _load_or_create_token(path: Path, env_name: str) -> tuple[str, str]:
    """Load a token from the environment or a mode-0600 local token file."""
    env_token = os.environ.get(env_name, "").strip()
    if env_token:
        if len(env_token) < MIN_TOKEN_LENGTH:
            raise ValueError(f"{env_name} must contain at least {MIN_TOKEN_LENGTH} characters")
        return env_token, f"environment variable {env_name}"

    path = path.expanduser().resolve()
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if len(token) < MIN_TOKEN_LENGTH:
            raise ValueError(f"token file {path} must contain at least {MIN_TOKEN_LENGTH} characters")
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return token, str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
    return token, str(path)


def _parse_episode_body(body: bytes) -> int:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("body must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"index"}:
        raise ValueError('body must be exactly {"index": <non-negative integer>}')
    index = payload["index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    return index


def _validate_backend_url(raw_url: str) -> str:
    url = raw_url.rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("OpenArmSim backend URL must use HTTP on the local loopback interface")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("OpenArmSim backend URL must not contain credentials, query, or fragment")
    return url


def _allowed_networks(cidrs: list[str]) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    try:
        return tuple(ipaddress.ip_network(cidr, strict=False) for cidr in cidrs)
    except ValueError as exc:
        raise ValueError(f"invalid --allow-cidr: {exc}") from exc


def _client_allowed(address: str, networks: tuple) -> bool:
    try:
        client = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(client in network for network in networks if client.version == network.version)


def _forward_episode(backend_url: str, index: int, timeout: float) -> tuple[int, dict]:
    endpoint = f"{backend_url}/api/episode?{urlencode({'index': index})}"
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request(endpoint, headers={"Accept": "application/json"}), timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {"ok": False, "error": f"OpenArmSim returned HTTP {exc.code}"}
        return exc.code, payload
    except URLError as exc:
        raise ConnectionError(f"OpenArmSim backend unavailable: {exc.reason}") from exc


class EpisodeGateway(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler, *, token, backend_url, timeout, allowed_networks):
        super().__init__(server_address, handler)
        self.token = token
        self.backend_url = backend_url
        self.backend_timeout = timeout
        self.allowed_networks = allowed_networks


class EpisodeGatewayHandler(BaseHTTPRequestHandler):
    server: EpisodeGateway
    server_version = "OpenArmEpisodeGateway/1.0"
    sys_version = ""

    def _send_json(self, status: HTTPStatus | int, payload: dict) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        supplied = header[len(prefix) :] if header.startswith(prefix) else ""
        return hmac.compare_digest(supplied.encode("utf-8"), self.server.token.encode("utf-8"))

    def _allow_client(self) -> bool:
        return _client_allowed(self.client_address[0], self.server.allowed_networks)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/healthz":
            self._send_json(HTTPStatus.OK, {"ok": True})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/episode":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        if not self._allow_client():
            self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "client network not allowed"})
            return
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "invalid bearer token"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY_BYTES:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid content length"})
            return
        try:
            index = _parse_episode_body(self.rfile.read(length))
            status, payload = _forward_episode(
                self.server.backend_url,
                index,
                self.server.backend_timeout,
            )
        except (ValueError, ConnectionError) as exc:
            status = HTTPStatus.BAD_GATEWAY if isinstance(exc, ConnectionError) else HTTPStatus.BAD_REQUEST
            self._send_json(status, {"ok": False, "error": str(exc)})
            return
        self._send_json(status, payload)

    def log_message(self, format: str, *args) -> None:
        print(f"gateway client={self.client_address[0]} {format % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Authenticated LAN gateway for OpenArmSim episode selection")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--openarm-url", default="http://127.0.0.1:8080")
    parser.add_argument("--backend-timeout", type=float, default=2.0)
    parser.add_argument("--token-env", default="OPENARM_SIM_GATEWAY_TOKEN")
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument(
        "--allow-cidr",
        action="append",
        default=None,
        help="allowed client subnet; repeat as needed (default: loopback + RFC1918 private IPv4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        token, token_source = _load_or_create_token(args.token_file, args.token_env)
        backend_url = _validate_backend_url(args.openarm_url)
        networks = _allowed_networks(args.allow_cidr or list(DEFAULT_ALLOWED_CIDRS))
        server = EpisodeGateway(
            (args.host, args.port),
            EpisodeGatewayHandler,
            token=token,
            backend_url=backend_url,
            timeout=args.backend_timeout,
            allowed_networks=networks,
        )
    except (OSError, ValueError) as exc:
        print(f"gateway startup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(
        f"OpenArm episode gateway serving on http://{args.host}:{args.port}/episode\n"
        f"backend={backend_url} token_source={token_source}\n"
        f"allowed_cidrs={','.join(str(network) for network in networks)}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
