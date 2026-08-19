"""ROS Int32 bridge for selecting OpenArmSim recorded episodes.

The ROS process deliberately stays separate from the MuJoCo/OpenPI virtual
environment.  It subscribes to ``std_msgs/Int32`` and forwards the zero-based
episode index to OpenArmSim's small HTTP control endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

DEFAULT_TOPIC = "/openarm_sim/episode"
DEFAULT_SERVER_URL = "http://127.0.0.1:8080"


def select_episode(server_url: str, index: int, *, timeout: float = 2.0) -> dict:
    """Request playback of one zero-based recorded episode index."""
    if index < 0:
        raise ValueError(f"episode index must be non-negative, got {index}")
    endpoint = f"{server_url.rstrip('/')}/api/episode?{urlencode({'index': index})}"
    request = Request(endpoint, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "OpenArmSim rejected episode selection"))
    return payload


def _run_ros1(args: argparse.Namespace) -> None:
    import rospy  # type: ignore[import-not-found]
    from std_msgs.msg import Int32  # type: ignore[import-not-found]

    rospy.init_node(args.node_name, anonymous=False)

    def callback(message) -> None:
        index = int(message.data)
        try:
            select_episode(args.server_url, index, timeout=args.timeout)
            rospy.loginfo("OpenArmSim selected recorded episode %d", index)
        except Exception as exc:  # keep subscriber alive across bad indexes/server restarts
            rospy.logerr("OpenArmSim episode %d selection failed: %s", index, exc)

    rospy.Subscriber(args.topic, Int32, callback, queue_size=1)
    rospy.loginfo(
        "OpenArmSim ROS1 bridge: topic=%s server=%s (zero-based index)",
        args.topic,
        args.server_url,
    )
    rospy.spin()


def _run_ros2(args: argparse.Namespace) -> None:
    import rclpy  # type: ignore[import-not-found]
    from rclpy.node import Node  # type: ignore[import-not-found]
    from std_msgs.msg import Int32  # type: ignore[import-not-found]

    class EpisodeBridge(Node):
        def __init__(self) -> None:
            super().__init__(args.node_name)
            self.subscription = self.create_subscription(Int32, args.topic, self._callback, 1)
            self.get_logger().info(
                f"OpenArmSim ROS2 bridge: topic={args.topic} "
                f"server={args.server_url} (zero-based index)"
            )

        def _callback(self, message: Int32) -> None:
            index = int(message.data)
            try:
                select_episode(args.server_url, index, timeout=args.timeout)
                self.get_logger().info(f"OpenArmSim selected recorded episode {index}")
            except Exception as exc:  # keep subscriber alive across bad indexes/server restarts
                self.get_logger().error(f"OpenArmSim episode {index} selection failed: {exc}")

    rclpy.init(args=None)
    node = EpisodeBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _auto_ros_version() -> str:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        try:
            import rospy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "neither ROS2 rclpy nor ROS1 rospy is importable; source the ROS environment "
                "and run this bridge with its ROS Python interpreter"
            ) from exc
        return "1"
    return "2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS Int32 to OpenArmSim episode bridge")
    parser.add_argument("--ros-version", choices=("auto", "1", "2"), default="auto")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--node-name", default="openarm_sim_episode_bridge")
    parser.add_argument("--timeout", type=float, default=2.0)
    # ROS1 may append remapping arguments such as ``__name:=...``.
    args, _ = parser.parse_known_args(argv)
    return args


def main() -> None:
    args = parse_args()
    version = _auto_ros_version() if args.ros_version == "auto" else args.ros_version
    try:
        if version == "1":
            _run_ros1(args)
        else:
            _run_ros2(args)
    except (ImportError, RuntimeError) as exc:
        print(f"ROS bridge startup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
