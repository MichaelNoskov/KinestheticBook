from __future__ import annotations

import asyncio
import csv
import math
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pygame

from core.controller import ESP32Controller

DATA_ROOT = Path(__file__).resolve().parent / "data"
CSV_HEADER = ("t_mono", "pitch", "roll", "yaw", "dp", "dr", "dy")
Row = tuple[float, float, float, float, float, float, float]

COL_BG = (22, 24, 32)
COL_PANEL = (34, 38, 48)
COL_PANEL_EDGE = (56, 62, 76)
COL_ACCENT = (72, 132, 204)
COL_ROW_SEL = (48, 72, 108)
COL_TEXT = (232, 234, 238)
COL_TEXT_MUTED = (148, 154, 168)
COL_SUCCESS = (88, 180, 120)
COL_DANGER = (200, 96, 96)
COL_WARN = (210, 170, 80)
COL_BTN = (58, 66, 86)
COL_BTN_EDGE = (88, 96, 118)
COL_BTN_DIS = (42, 44, 54)
COL_RECORD = (200, 72, 72)

WIN_W, WIN_H = 1024, 780
M = 14
ROW_H = 28
LIST_PAD = 10
GAP_S = 10
GAP_M = 18
GAP_L = 28


def sanitize_gesture_name(name: str) -> str | None:
    s = name.strip()
    if not s:
        return None
    s = re.sub(r'[/\\:\x00-\x1f]', "", s)
    s = s.strip(". ")
    return s or None


def list_gestures() -> list[tuple[str, int]]:
    if not DATA_ROOT.is_dir():
        return []
    out: list[tuple[str, int]] = []
    for p in sorted(DATA_ROOT.iterdir()):
        if p.is_dir():
            n = sum(1 for c in p.iterdir() if c.is_file() and c.suffix.lower() == ".csv")
            out.append((p.name, n))
    return out


def atomic_save_gesture_csv(gesture_dir: Path, rows: list[Row]) -> Path:
    gesture_dir.mkdir(parents=True, exist_ok=True)
    final_name = f"{time.time_ns()}.csv"
    final_path = gesture_dir / final_name
    tmp_path = gesture_dir / f"{final_name}.{os.getpid()}.tmp"
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def _ble_async_main(
    device_name: str,
    gyro_queue: queue.Queue,
    stop_evt: threading.Event,
):
    out = gyro_queue

    async def run():
        esp: ESP32Controller | None = None

        def gyro_cb(t_mono: float, pitch: float, roll: float, yaw: float) -> None:
            out.put(("gyro", t_mono, pitch, roll, yaw))

        try:
            esp = ESP32Controller(name=device_name, gyro_callback=gyro_cb)
            await esp.connect()
            out.put(("connected", True))
            while not stop_evt.is_set() and esp.client and esp.client.is_connected:
                await asyncio.sleep(0.05)
        except Exception as e:
            out.put(("error", str(e)))
        finally:
            if esp is not None and esp.client is not None:
                try:
                    await esp.disconnect()
                except Exception:
                    pass
            out.put(("connected", False))

    asyncio.run(run())


def _draw_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    font_title: pygame.font.Font,
    font_hint: pygame.font.Font | None = None,
    subtitle: str | None = None,
) -> pygame.Rect:
    pygame.draw.rect(surf, COL_PANEL, rect, border_radius=10)
    pygame.draw.rect(surf, COL_PANEL_EDGE, rect, 1, border_radius=10)
    ty = rect.y + 10
    surf.blit(font_title.render(title, True, COL_TEXT), (rect.x + 12, ty))
    ty += font_title.get_height() + 2
    if subtitle and font_hint:
        surf.blit(font_hint.render(subtitle, True, COL_TEXT_MUTED), (rect.x + 12, ty))
        ty += font_hint.get_height() + 4
    elif subtitle:
        ty += 4
    return pygame.Rect(rect.x + 12, ty, rect.w - 24, rect.bottom - ty - 10)


def _draw_labeled_value(
    surf: pygame.Surface,
    x: int,
    y: int,
    label: str,
    value: str,
    font_label: pygame.font.Font,
    font_value: pygame.font.Font,
    col_w: int,
    row_gap: int = 10,
) -> int:
    surf.blit(font_label.render(label, True, COL_TEXT_MUTED), (x, y))
    vs = font_value.render(value, True, COL_TEXT)
    surf.blit(vs, (x + col_w - vs.get_width(), y))
    return y + max(font_label.get_height(), font_value.get_height()) + row_gap


class App:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Запись жестов")
        self.font_title = pygame.font.SysFont("liberation sans", 22, bold=True)
        self.font = pygame.font.SysFont("liberation sans", 17)
        self.small = pygame.font.SysFont("liberation sans", 14)
        self.clock = pygame.time.Clock()

        self.device_name = "ESP32-C3-Motion"
        self.connected = False
        self.ble_busy = False
        self.user_disconnect = False
        self.ble_error: str | None = None

        self.stop_ble = threading.Event()
        self.ble_thread: threading.Thread | None = None
        self.msg_queue: queue.Queue = queue.Queue()

        self.gestures = list_gestures()
        self.selected: str | None = None
        self.list_scroll = 0

        self.recording = False
        self.record_buffer: list[Row] = []
        self.record_prev: tuple[float, float, float, float] | None = None

        self.input_new_name = False
        self.name_buffer = ""

        self.last_pitch = self.last_roll = self.last_yaw = 0.0

        self.btn_name_ok: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self.btn_name_cancel: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self.toast_text: str | None = None
        self.toast_kind: str = "info"
        self.toast_until: float = 0.0

        self._layout()

    def _toast(self, kind: str, text: str, seconds: float = 0.0) -> None:
        self.toast_kind = kind
        self.toast_text = text
        self.toast_until = time.monotonic() + seconds if seconds > 0 else 0.0

    def _toast_tick(self) -> None:
        if self.toast_until and time.monotonic() >= self.toast_until:
            self.toast_text = None
            self.toast_until = 0.0

    def _layout(self) -> None:
        title_h = 40
        y0 = M + title_h
        col_l = 278
        col_r = 300
        gap = GAP_M
        mid_x = M + col_l + gap
        mid_w = WIN_W - M * 2 - col_l - gap - col_r - gap
        bottom_reserve = 112

        self.title_rect = pygame.Rect(M, M, WIN_W - 2 * M, title_h)
        self.panel_left = pygame.Rect(M, y0, col_l, 300)
        self.panel_mid = pygame.Rect(mid_x, y0, mid_w, WIN_H - y0 - M - 8)
        self.panel_right = pygame.Rect(mid_x + mid_w + gap, y0, col_r, self.panel_mid.h)

        px = self.panel_left.x + 12
        py = self.panel_left.y + 46
        self.device_line_rect = pygame.Rect(px, py, self.panel_left.w - 24, 40)
        py = self.device_line_rect.bottom + GAP_M
        self.btn_connect = pygame.Rect(px, py, 120, 36)
        self.btn_disconnect = pygame.Rect(self.btn_connect.right + GAP_S, py, 120, 36)
        py = self.btn_connect.bottom + GAP_M
        self.status_dot_rect = pygame.Rect(px, py + 8, 10, 10)

        mid_top = self.panel_mid.y + 54
        self.btn_new = pygame.Rect(self.panel_mid.x + 12, mid_top, 140, 36)
        list_top = self.btn_new.bottom + GAP_M
        list_bottom = self.panel_mid.bottom - bottom_reserve
        self.list_rect = pygame.Rect(
            self.panel_mid.x + 12,
            list_top,
            self.panel_mid.w - 24,
            max(120, list_bottom - list_top),
        )
        self.input_rect = pygame.Rect(
            self.panel_mid.x + 12,
            self.panel_mid.bottom - bottom_reserve + 8,
            self.panel_mid.w - 24,
            40,
        )
        ny = self.input_rect.bottom + GAP_S
        self.btn_name_ok = pygame.Rect(self.panel_mid.x + 12, ny, 100, 34)
        self.btn_name_cancel = pygame.Rect(self.btn_name_ok.right + GAP_S, ny, 100, 34)

        ry = self.panel_right.y + 56
        self.rec_target_rect = pygame.Rect(self.panel_right.x + 12, ry, self.panel_right.w - 24, 32)
        ry = self.rec_target_rect.bottom + GAP_M
        self.btn_start = pygame.Rect(self.panel_right.x + 12, ry, (self.panel_right.w - 24 - GAP_S) // 2, 38)
        self.btn_stop = pygame.Rect(self.btn_start.right + GAP_S, ry, self.btn_start.w, 38)
        ry = self.btn_start.bottom + GAP_M
        self.rec_badge_rect = pygame.Rect(self.panel_right.x + 12, ry, self.panel_right.w - 24, 36)

        toast_h = 52
        toast_margin = 16
        self.toast_rect = pygame.Rect(
            self.panel_right.x + 12,
            self.panel_right.bottom - toast_h - toast_margin,
            self.panel_right.w - 24,
            toast_h,
        )
        sensor_top = self.rec_badge_rect.bottom + GAP_L
        room = max(0, self.toast_rect.y - GAP_M - sensor_top)
        sensor_h = max(88, min(200, room)) if room >= 88 else max(72, room)
        self.sensor_rect = pygame.Rect(
            self.panel_right.x + 12,
            sensor_top,
            self.panel_right.w - 24,
            sensor_h,
        )

    def _start_ble(self) -> None:
        if self.ble_thread and self.ble_thread.is_alive():
            return
        self.stop_ble.clear()
        self.ble_error = None
        self.user_disconnect = False
        self.ble_busy = True
        self.toast_text = None
        self.ble_thread = threading.Thread(
            target=_ble_async_main,
            args=(self.device_name, self.msg_queue, self.stop_ble),
            daemon=True,
        )
        self.ble_thread.start()

    def _stop_ble(self) -> None:
        self.user_disconnect = True
        self.stop_ble.set()
        self.ble_busy = True

    def _drain_queue(self) -> None:
        while True:
            try:
                kind, *rest = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "connected":
                self.connected = bool(rest[0])
                if self.connected:
                    self.ble_busy = False
                    self.ble_error = None
                    self._toast("success", "Подключено", 2.5)
                else:
                    self.ble_busy = self.ble_thread is not None and self.ble_thread.is_alive()
                    if self.user_disconnect:
                        self._toast("info", "Выкл.", 2.0)
                        self.user_disconnect = False
                    elif self.ble_error:
                        self._toast("error", self.ble_error, 0.0)
                    else:
                        self._toast("info", "Обрыв", 2.5)
                    self.recording = False
                    self.record_buffer.clear()
                    self.record_prev = None
            elif kind == "gyro":
                t_mono, p, r, y = rest
                self.last_pitch, self.last_roll, self.last_yaw = p, r, y
                if not self.connected:
                    self.connected = True
                    self.ble_busy = False
                    self.ble_error = None
                if not self.recording:
                    continue
                if self.record_prev is None:
                    dp = dr = dy = 0.0
                else:
                    _, pp, pr, py = self.record_prev
                    dp, dr, dy = p - pp, r - pr, y - py
                self.record_prev = (t_mono, p, r, y)
                self.record_buffer.append((t_mono, p, r, y, dp, dr, dy))
            elif kind == "error":
                self.ble_error = str(rest[0])
                self.ble_busy = False
                self._toast("error", self.ble_error, 0.0)

    def _gesture_dir(self, name: str) -> Path:
        return DATA_ROOT / name

    def _create_gesture(self, raw: str) -> None:
        name = sanitize_gesture_name(raw)
        if not name:
            self._toast("error", "Плохое имя", 0.0)
            return
        self._gesture_dir(name).mkdir(parents=True, exist_ok=True)
        self.selected = name
        self.gestures = list_gestures()
        self.input_new_name = False
        self.name_buffer = ""
        self._toast("success", name, 2.0)

    def _start_recording(self) -> None:
        if not self.connected or not self.selected:
            if not self.connected:
                self._toast("error", "Нет связи", 2.5)
            elif not self.selected:
                self._toast("error", "Нет жеста", 2.5)
            return
        self.record_buffer.clear()
        self.record_prev = None
        self.recording = True
        self.toast_text = None

    def _stop_recording(self) -> None:
        if not self.recording or not self.selected:
            self.recording = False
            return
        rows = list(self.record_buffer)
        self.recording = False
        self.record_buffer.clear()
        self.record_prev = None
        if not rows:
            self._toast("error", "Пусто", 3.0)
            return
        path = atomic_save_gesture_csv(self._gesture_dir(self.selected), rows)
        self.gestures = list_gestures()
        self._toast("success", path.name, 3.0)

    def _list_max_scroll(self) -> int:
        inner_h = self.list_rect.h - LIST_PAD * 2
        content_h = len(self.gestures) * ROW_H + LIST_PAD
        return max(0, content_h - inner_h)

    def _clamp_scroll(self) -> None:
        self.list_scroll = max(0, min(self.list_scroll, self._list_max_scroll()))

    def _click_list(self, pos: tuple[int, int]) -> None:
        if not self.list_rect.collidepoint(pos):
            return
        _mx, my = pos
        rel_y = my - self.list_rect.y - LIST_PAD + self.list_scroll
        idx = rel_y // ROW_H
        if 0 <= idx < len(self.gestures):
            self.selected = self.gestures[idx][0]

    def _draw_button(self, rect: pygame.Rect, label: str, enabled: bool = True) -> None:
        bg = COL_BTN if enabled else COL_BTN_DIS
        edge = COL_BTN_EDGE if enabled else COL_PANEL_EDGE
        pygame.draw.rect(self.screen, bg, rect, border_radius=8)
        pygame.draw.rect(self.screen, edge, rect, 1, border_radius=8)
        col = COL_TEXT if enabled else COL_TEXT_MUTED
        t = self.font.render(label, True, col)
        self.screen.blit(t, (rect.x + (rect.w - t.get_width()) // 2, rect.y + (rect.h - t.get_height()) // 2))

    def _connection_ui(self) -> tuple[pygame.Color, str]:
        if self.connected:
            return pygame.Color(COL_SUCCESS), "Ок"
        if self.ble_error and not self.ble_busy:
            return pygame.Color(COL_DANGER), "Сбой"
        if self.ble_busy:
            if self.user_disconnect:
                return pygame.Color(COL_WARN), "Отключение..."
            return pygame.Color(COL_WARN), "Подключение…"
        return pygame.Color(90, 94, 104), "Отключено"

    def _draw_truncated(self, text: str, font: pygame.font.Font, max_w: int, color: pygame.Color) -> pygame.Surface:
        if font.size(text)[0] <= max_w:
            return font.render(text, True, color)
        ell = "…"
        while text and font.size(text + ell)[0] > max_w:
            text = text[:-1]
        return font.render(text + ell, True, color)

    def run(self) -> None:
        running = True
        while running:
            self._drain_queue()
            self._toast_tick()
            if self.ble_thread and not self.ble_thread.is_alive():
                self.ble_busy = False

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 1:
                        if self.btn_connect.collidepoint(ev.pos) and not self.connected:
                            can = self.ble_thread is None or not self.ble_thread.is_alive()
                            if can:
                                self._start_ble()
                        elif self.btn_disconnect.collidepoint(ev.pos) and self.connected:
                            self._stop_ble()
                        elif self.btn_new.collidepoint(ev.pos):
                            self.input_new_name = True
                            self.name_buffer = ""
                        elif self.btn_start.collidepoint(ev.pos):
                            self._start_recording()
                        elif self.btn_stop.collidepoint(ev.pos):
                            self._stop_recording()
                        elif self.list_rect.collidepoint(ev.pos):
                            self._click_list(ev.pos)
                        elif self.input_new_name and self.btn_name_ok.collidepoint(ev.pos):
                            self._create_gesture(self.name_buffer)
                        elif self.input_new_name and self.btn_name_cancel.collidepoint(ev.pos):
                            self.input_new_name = False
                            self.name_buffer = ""
                elif ev.type == pygame.MOUSEWHEEL:
                    if self.list_rect.collidepoint(pygame.mouse.get_pos()):
                        self.list_scroll = max(0, self.list_scroll - ev.y * ROW_H)
                        self._clamp_scroll()
                elif ev.type == pygame.KEYDOWN and self.input_new_name:
                    if ev.key == pygame.K_BACKSPACE:
                        self.name_buffer = self.name_buffer[:-1]
                    elif ev.unicode and ev.unicode.isprintable():
                        self.name_buffer += ev.unicode

            self._clamp_scroll()

            self.screen.fill(COL_BG)

            _draw_panel(self.screen, self.panel_left, "Связь", self.font_title, self.small, None)
            dev_surf = self._draw_truncated(
                self.device_name, self.small, self.device_line_rect.w, pygame.Color(COL_TEXT)
            )
            self.screen.blit(dev_surf, (self.device_line_rect.x, self.device_line_rect.y + 10))
            self._draw_button(self.btn_connect, "Подключить", not self.connected and (self.ble_thread is None or not self.ble_thread.is_alive()))
            self._draw_button(self.btn_disconnect, "Отключить", self.connected)
            dot_col, link_caption = self._connection_ui()
            pygame.draw.circle(self.screen, dot_col, self.status_dot_rect.center, 5)
            cap = self.small.render(link_caption, True, COL_TEXT)
            self.screen.blit(cap, (self.status_dot_rect.right + 10, self.status_dot_rect.y - 2))

            _draw_panel(self.screen, self.panel_mid, "Датасет", self.font_title, self.small, None)
            self._draw_button(self.btn_new, "новый жест", True)

            inner = self.list_rect.inflate(-LIST_PAD * 2, -LIST_PAD * 2)
            inner.x = self.list_rect.x + LIST_PAD
            inner.y = self.list_rect.y + LIST_PAD
            pygame.draw.rect(self.screen, (28, 30, 38), self.list_rect, border_radius=8)
            pygame.draw.rect(self.screen, COL_PANEL_EDGE, self.list_rect, 1, border_radius=8)
            clip = self.screen.get_clip()
            self.screen.set_clip(self.list_rect)
            y0 = inner.y - self.list_scroll
            name_max = inner.w - 56
            for i, (name, cnt) in enumerate(self.gestures):
                y = y0 + i * ROW_H
                if y + ROW_H < self.list_rect.top or y > self.list_rect.bottom:
                    continue
                row_rect = pygame.Rect(inner.x, y, inner.w, ROW_H - 2)
                if name == self.selected:
                    pygame.draw.rect(self.screen, COL_ROW_SEL, row_rect, border_radius=6)
                name_s = self._draw_truncated(name, self.font, name_max, pygame.Color(COL_TEXT))
                self.screen.blit(name_s, (row_rect.x + 8, row_rect.y + 4))
                cnt_s = self.small.render(str(cnt), True, COL_TEXT_MUTED)
                self.screen.blit(cnt_s, (row_rect.right - cnt_s.get_width() - 10, row_rect.y + 6))
            self.screen.set_clip(clip)

            _draw_panel(self.screen, self.panel_right, "Запись", self.font_title, self.small, None)
            target = self.selected or "—"
            ts = self.font.render(target, True, COL_ACCENT if self.selected else COL_TEXT_MUTED)
            self.screen.blit(ts, (self.rec_target_rect.x, self.rec_target_rect.y + 6))

            rec_ok = self.connected and self.selected and not self.recording
            stop_ok = self.recording
            self._draw_button(self.btn_start, "Старт", rec_ok)
            self._draw_button(self.btn_stop, "Стоп", stop_ok)

            if self.recording:
                pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 6.0)
                bar_c = (
                    int(COL_RECORD[0]),
                    int(COL_RECORD[1] * (0.65 + 0.35 * pulse)),
                    int(COL_RECORD[2]),
                )
                pygame.draw.rect(self.screen, bar_c, self.rec_badge_rect, border_radius=6)
                lab = self.small.render("Запись", True, COL_TEXT)
                self.screen.blit(
                    lab,
                    (
                        self.rec_badge_rect.x + (self.rec_badge_rect.w - lab.get_width()) // 2,
                        self.rec_badge_rect.y + (self.rec_badge_rect.h - lab.get_height()) // 2,
                    ),
                )
            else:
                pygame.draw.rect(self.screen, (32, 36, 44), self.rec_badge_rect, border_radius=6)
                pygame.draw.rect(self.screen, COL_PANEL_EDGE, self.rec_badge_rect, 1, border_radius=6)
                idle = self.small.render("Пауза", True, COL_TEXT_MUTED)
                self.screen.blit(
                    idle,
                    (
                        self.rec_badge_rect.x + (self.rec_badge_rect.w - idle.get_width()) // 2,
                        self.rec_badge_rect.y + (self.rec_badge_rect.h - idle.get_height()) // 2,
                    ),
                )

            pygame.draw.rect(self.screen, (26, 28, 36), self.sensor_rect, border_radius=8)
            pygame.draw.rect(self.screen, COL_PANEL_EDGE, self.sensor_rect, 1, border_radius=8)
            self.screen.blit(
                self.font.render("Гироскоп", True, COL_TEXT),
                (self.sensor_rect.x + 14, self.sensor_rect.y + 12),
            )
            vx = self.sensor_rect.x + 14
            vy = self.sensor_rect.y + 46
            col_w = self.sensor_rect.w - 28
            vy = _draw_labeled_value(
                self.screen,
                vx,
                vy,
                "pitch",
                f"{self.last_pitch:+.1f}°",
                self.small,
                self.font,
                col_w,
                12,
            )
            vy = _draw_labeled_value(
                self.screen,
                vx,
                vy,
                "roll",
                f"{self.last_roll:+.1f}°",
                self.small,
                self.font,
                col_w,
                12,
            )
            _draw_labeled_value(
                self.screen,
                vx,
                vy,
                "yaw",
                f"{self.last_yaw:+.1f}°",
                self.small,
                self.font,
                col_w,
                12,
            )

            if self.toast_text:
                tc = COL_TEXT
                if self.toast_kind == "error":
                    tc = COL_DANGER
                elif self.toast_kind == "success":
                    tc = COL_SUCCESS
                elif self.toast_kind == "info":
                    tc = COL_WARN
                tw = self.toast_rect.w - 16
                surf = self._draw_truncated(self.toast_text, self.font, tw, pygame.Color(tc))
                ty = self.toast_rect.y + (self.toast_rect.h - surf.get_height()) // 2
                self.screen.blit(surf, (self.toast_rect.x + 8, ty))

            pygame.draw.rect(self.screen, (30, 32, 40), self.input_rect, border_radius=8)
            pygame.draw.rect(self.screen, COL_PANEL_EDGE, self.input_rect, 1, border_radius=8)
            if self.input_new_name:
                prompt = self.small.render("Имя", True, COL_TEXT_MUTED)
                self.screen.blit(prompt, (self.input_rect.x, self.input_rect.y - 22))
                cur = self.name_buffer + ("|" if int(time.monotonic() * 2) % 2 else "")
                it = self.font.render(cur, True, COL_TEXT)
                self._draw_button(self.btn_name_ok, "Ок", True)
                self._draw_button(self.btn_name_cancel, "Отмена", True)
                self.screen.blit(it, (self.input_rect.x + 10, self.input_rect.y + (self.input_rect.h - it.get_height()) // 2))

            pygame.display.flip()
            self.clock.tick(30)

        self._stop_ble()
        if self.ble_thread is not None:
            self.ble_thread.join(timeout=3.0)
        pygame.quit()


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    App().run()


if __name__ == "__main__":
    main()
