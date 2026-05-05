import asyncio
from bleak import BleakScanner, BleakClient

SERVICE_UUID = "6E400001-B5A3-F393-E0E9-E50E24DCCA9E"
RX_CHAR_UUID = "6E400002-B5A3-F393-E0E9-E50E24DCCA9E"
TX_CHAR_UUID = "6E400003-B5A3-F393-E0E9-E50E24DCCA9E"

GYRO_INTERVAL_MS = 50          # 0 – выключить, иначе интервал в мс
PING_INTERVAL = 5.0            # секунды


class ESP32Controller:
    def __init__(self, name="ESP32-C3-Motion"):
        self.name = name
        self.client = None
        self.num_motors = None
        self.response_event = asyncio.Event()
        self.last_response = ""
        self.command_counter = 0
        self.last_gyro = None          # (pitch, roll, yaw)

    def _on_notify(self, sender, data):
        msg = data.decode().strip()
        print(f"[ESP32] {msg}")
        if msg.startswith("MOTORS="):
            self.last_response = msg
            self.num_motors = int(msg.split("=")[1])
            self.response_event.set()
        elif msg.startswith("GYRO,"):
            parts = msg.split(",")
            if len(parts) == 4:
                _, p, r, y = parts
                self.last_gyro = (float(p), float(r), float(y))
        elif msg.startswith(("RECEIVED:", "QUEUE_FULL:", "DONE:", "INVALID:")):
            self.last_response = msg
            self.response_event.set()

    async def _wait_for_motors(self):
        while True:
            await self.response_event.wait()
            self.response_event.clear()
            if self.last_response.startswith("MOTORS="):
                self.num_motors = int(self.last_response.split("=")[1])
                return

    async def request_motors_count(self, timeout=5.0):
        await self.client.write_gatt_char(RX_CHAR_UUID, b"MOTORS", response=True)
        try:
            await asyncio.wait_for(self._wait_for_motors(), timeout=timeout)
        except asyncio.TimeoutError:
            raise Exception("No MOTORS response")

    async def connect(self):
        print(f"Scanning for '{self.name}'...")
        devices = await BleakScanner.discover()
        target = None
        for d in devices:
            if d.name and self.name.lower() in d.name.lower():
                target = d
                break
        if not target:
            raise Exception(f"Device '{self.name}' not found")

        self.client = BleakClient(target.address)
        await self.client.connect()
        await self.client.start_notify(TX_CHAR_UUID, self._on_notify)
        await asyncio.sleep(0.2)   # пауза для подписки

        # Запрашиваем количество моторов
        await self.request_motors_count()
        print(f"Connected. Motors: {self.num_motors}")

        # Устанавливаем интервал получения данных с гироскопа
        await self.client.write_gatt_char(RX_CHAR_UUID, f"GYRO,{GYRO_INTERVAL_MS}".encode(), response=True)
        print(f"Set gyro interval: {GYRO_INTERVAL_MS} ms")

        asyncio.create_task(self._ping_loop())

    async def _ping_loop(self):
        while self.client and self.client.is_connected:
            await asyncio.sleep(PING_INTERVAL)
            if not self.client.is_connected:
                break
            self.command_counter += 1
            ping_id = self.command_counter
            await self.client.write_gatt_char(RX_CHAR_UUID, f"PING,{ping_id}".encode(), response=True)

    async def send_command(self, motor: int, duration_ms: int, max_retries=5):
        if not (1 <= motor <= self.num_motors):
            print(f"Motor {motor} out of range (1..{self.num_motors})")
            return False
        if duration_ms > 1000:
            print("Duration > 1000 ms not allowed (controller limit)")
            return False

        retries = 0
        while retries < max_retries:
            self.command_counter += 1
            cmd_id = self.command_counter
            cmd = f"{cmd_id},{motor},{duration_ms}"
            await self.client.write_gatt_char(RX_CHAR_UUID, cmd.encode(), response=True)

            try:
                await asyncio.wait_for(self.response_event.wait(), timeout=3.0)
                self.response_event.clear()
            except asyncio.TimeoutError:
                print(f"No response for {cmd_id}, retry {retries+1}")
                retries += 1
                continue

            resp = self.last_response
            if resp.startswith("INVALID:"):
                print(f"Command invalid: {resp}")
                return False
            elif resp.startswith("QUEUE_FULL:"):
                _, rest = resp.split(":")
                ret_id, wait_ms = rest.split(",")
                if int(ret_id) == cmd_id:
                    print(f"Queue full, waiting {wait_ms} ms")
                    await asyncio.sleep(int(wait_ms) / 1000.0)
                    retries += 1
                    continue
            elif resp.startswith("RECEIVED:"):
                try:
                    await asyncio.wait_for(self.response_event.wait(), timeout=duration_ms/1000 + 3.0)
                    self.response_event.clear()
                except asyncio.TimeoutError:
                    print(f"Timeout waiting DONE for {cmd_id}")
                    return False
                if self.last_response.startswith("DONE:"):
                    done_id = int(self.last_response.split(":")[1])
                    if done_id == cmd_id:
                        print(f"Command {cmd_id} executed successfully")
                        return True
                    else:
                        print(f"Wrong DONE id: {done_id}, expected {cmd_id}")
                        return False
                else:
                    print(f"Unexpected after RECEIVED: {self.last_response}")
                    return False
            else:
                print(f"Unexpected response: {resp}")
                return False

        print(f"Failed after {max_retries} retries")
        return False

    async def send_batch(self, commands, delay=0.1):
        for motor, duration in commands:
            await self.send_command(motor, duration)
            await asyncio.sleep(delay)

    async def disconnect(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("Disconnected")


async def roll_based_motors_loop(controller: ESP32Controller, duration_ms: int = 300):
    if controller.num_motors is None:
        print("Motors count unknown. Did you call connect()?")
        return

    num_motors = controller.num_motors
    total_range = 180.0          # от -90 до 90 = 180 градусов
    sector_size = total_range / num_motors
    print(f"Sector size: {sector_size:.1f}°, motors: {num_motors}")

    while controller.client and controller.client.is_connected:
        # Ожидание первого значения гироскопа
        while controller.last_gyro is None:
            await asyncio.sleep(0.05)
            if not controller.client.is_connected:
                return

        pitch, _, _ = controller.last_gyro   # (pitch, roll, yaw)

        # Диапазон pitch [-90..90], приводим к [0..180]
        pitch_shifted = pitch + 90.0
        sector_idx = int(pitch_shifted / sector_size)
        if sector_idx >= num_motors:
            sector_idx = num_motors - 1
        motor = sector_idx + 1

        print(f"Pitch: {pitch:.1f}° -> motor {motor} (sector {sector_idx+1}/{num_motors})")
        await controller.send_command(motor, duration_ms)


async def main():
    esp = ESP32Controller()
    try:
        await esp.connect()
        # await roll_based_motors_loop(esp, duration_ms=300)
        while True:
        # Пример: запускаем все моторы по очереди
            for m in range(6, int(esp.num_motors) + 1):
                print(f"\n--- Motor {m} ---")
                ok = await esp.send_command(m, 500)
                if not ok:
                    print("Stopping due to failure")
                    break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await esp.disconnect()


if __name__ == "__main__":
    asyncio.run(main())