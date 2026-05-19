import asyncio
import logging
import time
from typing import Callable, Optional

from bleak import BleakScanner, BleakClient

SERVICE_UUID = "6E400001-B5A3-F393-E0E9-E50E24DCCA9E"
RX_CHAR_UUID = "6E400002-B5A3-F393-E0E9-E50E24DCCA9E"
TX_CHAR_UUID = "6E400003-B5A3-F393-E0E9-E50E24DCCA9E"

PING_INTERVAL = 5.0

GyroCallback = Optional[Callable[[float, float, float, float], None]]

_log = logging.getLogger(__name__)


class CommandHandle:

    def __init__(self, cmd_id: int, q: "asyncio.Queue[str]", duration_ms: int):
        self._cmd_id = cmd_id
        self._q = q
        self._duration_ms = duration_ms

    async def wait_done(self, timeout: float | None = None) -> bool:
        if timeout is None:
            timeout = self._duration_ms / 1000 + 3.0
        try:
            msg = await asyncio.wait_for(self._q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            _log.warning("Timeout waiting DONE for cmd %d", self._cmd_id)
            return False
        if msg == "__DISCONNECTED__":
            return False
        if msg.startswith("DONE:"):
            return int(msg.split(":")[1]) == self._cmd_id
        _log.warning("Unexpected message in wait_done for cmd %d: %s", self._cmd_id, msg)
        return False


class ESP32Controller:
    def __init__(
        self,
        name: str = "ESP32-C3-Motion",
        gyro_callback: GyroCallback = None,
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[], None]] = None,
        auto_reconnect: bool = False,
    ):
        self.name = name
        self.gyro_callback = gyro_callback
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self.auto_reconnect = auto_reconnect

        self.client: Optional[BleakClient] = None
        self.num_motors: Optional[int] = None
        self.last_gyro: Optional[tuple[float, float, float]] = None

        self._cmd_counter = 0
        self._pending: dict[int, asyncio.Queue[str]] = {}
        self._motors_queue: asyncio.Queue[str] = asyncio.Queue()
        self._gyro_interval_ms: Optional[int] = None
        self._reconnecting = False
        self._shutdown = False

    @property
    def is_connected(self) -> bool:
        return bool(self.client and self.client.is_connected)

    def _next_id(self) -> int:
        self._cmd_counter = (self._cmd_counter % 65535) + 1
        return self._cmd_counter

    def _on_notify(self, _sender, data: bytes) -> None:
        msg = data.decode().strip()
        _log.debug("[ESP32] %s", msg)

        if msg.startswith("MOTORS="):
            self.num_motors = int(msg.split("=")[1])
            self._motors_queue.put_nowait(msg)

        elif msg.startswith("GYRO,"):
            parts = msg.split(",")
            if len(parts) == 4:
                _, p, r, y = parts
                pitch, roll, yaw = float(p), float(r), float(y)
                self.last_gyro = (pitch, roll, yaw)
                if self.gyro_callback is not None:
                    self.gyro_callback(time.monotonic(), pitch, roll, yaw)

        elif msg.startswith(("RECEIVED:", "DONE:", "QUEUE_FULL:", "INVALID:")):
            try:
                cmd_id = int(msg.split(":")[1].split(",")[0])
            except (IndexError, ValueError):
                _log.warning("Cannot parse ID from: %s", msg)
                return
            if cmd_id in self._pending:
                self._pending[cmd_id].put_nowait(msg)
                if msg.startswith(("DONE:", "INVALID:")):
                    del self._pending[cmd_id]
            else:
                _log.debug("No pending command for ID %d: %s", cmd_id, msg)

    def _on_ble_disconnect(self, _client) -> None:
        _log.warning("BLE disconnected")
        if self.on_disconnected:
            self.on_disconnected()
        for q in self._pending.values():
            q.put_nowait("__DISCONNECTED__")
        self._pending.clear()
        if self.auto_reconnect and not self._shutdown:
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._reconnect_loop())
            )

    async def _reconnect_loop(self) -> None:
        if self._reconnecting:
            return
        self._reconnecting = True
        delay = 2.0
        while not self._shutdown:
            _log.info("Reconnecting in %.0fs...", delay)
            await asyncio.sleep(delay)
            try:
                await self._do_connect()
                if self._gyro_interval_ms is not None:
                    await self.set_gyro_interval(self._gyro_interval_ms)
                _log.info("Reconnected. Motors: %d", self.num_motors)
                if self.on_connected:
                    self.on_connected()
                break
            except Exception as e:
                _log.warning("Reconnect failed: %s", e)
                delay = min(delay * 2, 30.0)
        self._reconnecting = False

    async def _do_connect(self) -> None:
        _log.info("Scanning for '%s'...", self.name)
        devices = await BleakScanner.discover()
        target = next(
            (d for d in devices if d.name and self.name.lower() in d.name.lower()),
            None,
        )
        if not target:
            raise RuntimeError(f"Device '{self.name}' not found")

        self.client = BleakClient(target.address, disconnected_callback=self._on_ble_disconnect)
        await self.client.connect()
        await self.client.start_notify(TX_CHAR_UUID, self._on_notify)
        await asyncio.sleep(0.2)

        await self._send_ping()
        await self.request_motors_count()

    async def connect(self) -> None:
        self._shutdown = False
        await self._do_connect()
        _log.info("Connected. Motors: %d", self.num_motors)
        if self.on_connected:
            self.on_connected()
        asyncio.create_task(self._ping_loop())

    async def disconnect(self) -> None:
        self._shutdown = True
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            _log.info("Disconnected")

    async def set_gyro_interval(self, interval_ms: int) -> None:
        if not 0 <= interval_ms <= 60000:
            raise ValueError("interval_ms must be in 0..60000")
        self._gyro_interval_ms = interval_ms
        await self.client.write_gatt_char(
            RX_CHAR_UUID, f"GYRO,{interval_ms}".encode(), response=True
        )

    async def request_motors_count(self, timeout: float = 5.0) -> None:
        await self.client.write_gatt_char(RX_CHAR_UUID, b"MOTORS", response=True)
        try:
            resp = await asyncio.wait_for(self._motors_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError("No MOTORS response")
        if not resp.startswith("MOTORS="):
            raise RuntimeError(f"Unexpected response to MOTORS: {resp}")

    async def send_command(
        self, motor: int, duration_ms: int, max_retries: int = 5
    ) -> Optional["CommandHandle"]:
        """Send a motor command and wait for RECEIVED: acknowledgment.

        Returns a CommandHandle on success (use wait_done() to await completion),
        or None if the command failed validation, was rejected, or retries exhausted.
        """
        if not self.is_connected:
            _log.warning("send_command called while disconnected")
            return None
        if not (1 <= motor <= self.num_motors):
            _log.warning("Motor %d out of range (1..%d)", motor, self.num_motors)
            return None
        if not (1 <= duration_ms <= 1000):
            _log.warning("duration_ms %d out of range (1..1000)", duration_ms)
            return None

        for attempt in range(max_retries):
            cmd_id = self._next_id()
            q: asyncio.Queue[str] = asyncio.Queue()
            self._pending[cmd_id] = q

            try:
                await self.client.write_gatt_char(
                    RX_CHAR_UUID, f"{cmd_id},{motor},{duration_ms}".encode(), response=True
                )
            except Exception as e:
                _log.warning("Write failed for cmd %d: %s", cmd_id, e)
                self._pending.pop(cmd_id, None)
                return None

            try:
                resp = await asyncio.wait_for(q.get(), timeout=3.0)
            except asyncio.TimeoutError:
                _log.warning("No response for cmd %d, attempt %d", cmd_id, attempt + 1)
                self._pending.pop(cmd_id, None)
                continue

            if resp == "__DISCONNECTED__":
                return None

            if resp.startswith("INVALID:"):
                _log.error("Command rejected as invalid: %s", resp)
                return None

            if resp.startswith("QUEUE_FULL:"):
                _, rest = resp.split(":")
                ret_id, wait_ms = rest.split(",")
                self._pending.pop(cmd_id, None)
                if int(ret_id) == cmd_id:
                    _log.debug("Queue full, waiting %s ms", wait_ms)
                    await asyncio.sleep(int(wait_ms) / 1000.0)
                continue

            if resp.startswith("RECEIVED:"):
                return CommandHandle(cmd_id, q, duration_ms)

            _log.warning("Unexpected response: %s", resp)
            self._pending.pop(cmd_id, None)
            return None

        _log.error("Failed after %d retries", max_retries)
        return None

    async def _send_ping(self) -> None:
        if not self.client or not self.client.is_connected:
            return
        await self.client.write_gatt_char(
            RX_CHAR_UUID, f"PING,{self._next_id()}".encode(), response=False
        )

    async def _ping_loop(self) -> None:
        while not self._shutdown and self.is_connected:
            await asyncio.sleep(PING_INTERVAL)
            if self.is_connected:
                await self._send_ping()
