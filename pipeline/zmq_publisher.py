"""ZMQ PUB socket for publishing robot commands.

Publishes RobotCommand JSON on tcp://*:5556 for the endoscope_control
subscriber to receive and execute on the robot.
"""

import json
import logging
import time

import zmq

from schema.command_schema import RobotCommand

logger = logging.getLogger(__name__)


class CommandPublisher:
    """Publishes RobotCommand messages over ZMQ PUB/SUB."""

    def __init__(self, port: int = 5556):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        # Slow joiner: subscribers need time to connect before first message
        time.sleep(0.5)
        logger.info("ZMQ publisher bound to tcp://*:%d", port)

    def publish(self, command: RobotCommand) -> str:
        """Publish a command as JSON over ZMQ.

        Args:
            command: The validated RobotCommand to publish.

        Returns:
            The JSON string that was sent.
        """
        data = command.to_zmq_dict()
        json_str = json.dumps(data)
        self.socket.send_string(json_str)
        logger.info("Published: %s", json_str)
        return json_str

    def close(self) -> None:
        """Clean up ZMQ resources."""
        self.socket.close()
        self.context.term()
        logger.info("ZMQ publisher closed.")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
