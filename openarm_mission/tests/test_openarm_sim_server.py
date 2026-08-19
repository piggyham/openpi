"""Unit tests for OpenArmSim runtime source selection helpers."""

from pathlib import Path
import tempfile
from unittest import mock

import pytest

from openarm_mission.openarm_sim.lan_episode_gateway import _allowed_networks
from openarm_mission.openarm_sim.lan_episode_gateway import _client_allowed
from openarm_mission.openarm_sim.lan_episode_gateway import _load_or_create_token
from openarm_mission.openarm_sim.lan_episode_gateway import _parse_episode_body
from openarm_mission.openarm_sim.lan_episode_gateway import _validate_backend_url
from openarm_mission.openarm_sim.playback import SimPlayback
from openarm_mission.openarm_sim.ros_episode_bridge import select_episode
from openarm_mission.openarm_sim.server import _recorded_specs_for_path
from openarm_mission.openarm_sim.server import _validate_live_url


class TestRuntimeSourceSelection:
    def test_recorded_reality_allows_dynamic_but_live_does_not(self) -> None:
        playback = SimPlayback.__new__(SimPlayback)
        playback._episode = mock.Mock(live=False)  # noqa: SLF001
        playback._mode = "kinematic"  # noqa: SLF001
        playback._k = 7  # noqa: SLF001
        playback.seek = mock.Mock()

        playback.set_mode("dynamic")
        assert playback.mode == "dynamic"
        playback.seek.assert_called_once_with(7)

        playback._episode = mock.Mock(live=True)  # noqa: SLF001
        with pytest.raises(ValueError, match="live Actual"):
            playback.set_mode("dynamic")

    def test_single_parquet_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parquet = Path(tmp) / "episode_000123.parquet"
            parquet.touch()
            path, specs, single_file = _recorded_specs_for_path(str(parquet))

        assert path == parquet.resolve()
        assert single_file
        assert specs == [("episode_000123", "lerobot", parquet.resolve())]

    def test_lerobot_directory_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "data" / "chunk-000" / "episode_000007.parquet"
            parquet.parent.mkdir(parents=True)
            parquet.touch()
            path, specs, single_file = _recorded_specs_for_path(str(root))

        assert path == root.resolve()
        assert not single_file
        assert specs == [("episode_000007", "lerobot", parquet.resolve())]

    def test_rejects_unsupported_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.npz"
            path.touch()
            with pytest.raises(ValueError, match="parquet"):
                _recorded_specs_for_path(str(path))

    def test_live_url_validation(self) -> None:
        assert (
            _validate_live_url(" http://127.0.0.1:9000/sse/sim ")
            == "http://127.0.0.1:9000/sse/sim"
        )
        with pytest.raises(ValueError, match="http"):
            _validate_live_url("127.0.0.1:9000/sse/sim")

    def test_ros_bridge_forwards_zero_based_index(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"ok": true, "index": 12}'
        with mock.patch(
            "openarm_mission.openarm_sim.ros_episode_bridge.urlopen",
            return_value=response,
        ) as open_url:
            payload = select_episode("http://127.0.0.1:8080/", 12)

        assert payload == {"ok": True, "index": 12}
        request = open_url.call_args.args[0]
        assert request.full_url == "http://127.0.0.1:8080/api/episode?index=12"

    def test_ros_bridge_rejects_negative_index(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            select_episode("http://127.0.0.1:8080", -1)

    def test_gateway_accepts_only_exact_integer_payload(self) -> None:
        assert _parse_episode_body(b'{"index": 7}') == 7
        for body in (b'{"index": -1}', b'{"index": true}', b'{"index": 1, "path": "/tmp"}', b"not json"):
            with pytest.raises(ValueError, match="body|index"):
                _parse_episode_body(body)

    def test_gateway_backend_is_loopback_only(self) -> None:
        assert _validate_backend_url("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"
        with pytest.raises(ValueError, match="loopback"):
            _validate_backend_url("http://192.168.1.20:8080")

    def test_gateway_private_network_allowlist(self) -> None:
        networks = _allowed_networks(["127.0.0.0/8", "192.168.0.0/16"])
        assert _client_allowed("127.0.0.1", networks)
        assert _client_allowed("192.168.12.4", networks)
        assert not _client_allowed("8.8.8.8", networks)

    def test_gateway_creates_private_token_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_OPENARM_GATEWAY_TOKEN", raising=False)
        with tempfile.TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "gateway_token"
            token, source = _load_or_create_token(token_path, "TEST_OPENARM_GATEWAY_TOKEN")
            mode = token_path.stat().st_mode & 0o777

        assert len(token) >= 32
        assert source == str(token_path.resolve())
        assert mode == 0o600
