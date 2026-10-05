"""Small OpenPI-compatible WebSocket client used by the Panda deployment example."""

from __future__ import annotations

import contextlib
from typing import Any


class Pi05WebSocketClient:
    """Exchange msgpack-numpy observations and actions with serve_policy.py."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        open_timeout: float = 10.0,
        response_timeout: float = 30.0,
    ) -> None:
        try:
            from websockets.sync.client import connect
            import pi05_msgpack_numpy as msgpack_numpy
        except ImportError as error:
            raise RuntimeError(
                "Pi05 networking requires websockets and msgpack; "
                "install examples/panda/requirements_pi05.txt"
            ) from error

        self._packer = msgpack_numpy.Packer()
        self._unpackb = msgpack_numpy.unpackb
        self._response_timeout = response_timeout
        self._connection = connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            ping_interval=None,
            open_timeout=open_timeout,
            close_timeout=5.0,
        )
        self.metadata = self._unpack(self._connection.recv(timeout=response_timeout))

    def _unpack(self, response: str | bytes) -> dict[str, Any]:
        if isinstance(response, str):
            raise RuntimeError(f"Policy server returned an error:\n{response}")
        result = self._unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"Policy server returned {type(result).__name__}, expected dict")
        return result

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._connection.send(self._packer.pack(observation))
        return self._unpack(self._connection.recv(timeout=self._response_timeout))

    def reset(self) -> dict[str, Any]:
        self._connection.send(self._packer.pack({"reset": True}))
        return self._unpack(self._connection.recv(timeout=self._response_timeout))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._connection.close()

    def __enter__(self) -> "Pi05WebSocketClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
