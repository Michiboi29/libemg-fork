import platform
import time
import threading
from multiprocessing import Event, Process
from queue import Queue, Empty
from typing import Callable

import numpy as np
import serial
import serial.tools.list_ports

from libemg.shared_memory_manager import SharedMemoryManager


# ============================================================
# EmagerPuck 8192-byte frames with packed 15-bit signed EMG
# ============================================================

class EmagerPuck:
    """
    Reader for the 32-channel EMaGer Puck frame format.

    Expected 8192-byte frame:
      [0]        0xAA
      [1]        0x55
      [EMG_START:EMG_END]    packed 15-bit signed EMG payload
      [CTR_START:CTR_START+4] frame counter, big-endian uint32
      [8190]     0x55
      [8191]     0xAA

    Active format from the working standalone code:
      EMG_START = 6
      EMG_LEN = 8160
      CTR_START = 2

    Each frame:
      8160 bytes * 8 bits / 15 bits = 4352 EMG values
      4352 values / 32 channels = 136 samples per channel
    """

    HDR0, HDR1 = 0xAA, 0x55
    TLR0, TLR1 = 0x55, 0xAA

    FRAME_SIZE = 8192

    EMG_START = 6
    EMG_LEN = 8160
    EMG_END = EMG_START + EMG_LEN

    CTR_START = 2

    TRAILER0_I = 8190
    TRAILER1_I = 8191

    CHANNELS = 32
    EMG_VALUES_PER_FRAME = 4352
    SAMPLES_PER_CH_PER_FRAME = EMG_VALUES_PER_FRAME // CHANNELS  # 136

    def __init__(
        self,
        baud_rate: int = 3000000,
        com_name=None,
        vid_pid=(12259, 256),
        restore_lsb0: bool = True,
        debug: bool = False,
    ):
        self.com_name = com_name
        self.vid_pid = vid_pid
        self.restore_lsb0 = bool(restore_lsb0)
        self.debug = bool(debug)

        ports = list(serial.tools.list_ports.comports())
        com_port = None

        for p in ports:
            if self.com_name is None:
                if (p.vid, p.pid) == self.vid_pid:
                    com_port = p.name if platform.system() == "Windows" else p.device.replace("cu", "tty")
                    break
            else:
                dev = getattr(p, "device", "") or ""
                name = getattr(p, "name", "") or ""
                desc = getattr(p, "description", "") or ""

                if self.com_name in dev or self.com_name in name or self.com_name in desc:
                    com_port = p.name if platform.system() == "Windows" else p.device.replace("cu", "tty")
                    break

        if com_port is None:
            ports_info = []
            for p in ports:
                dev = getattr(p, "device", None) or getattr(p, "name", None) or "<unknown>"
                desc = getattr(p, "description", "") or "<no description>"
                vid = getattr(p, "vid", None)
                pid = getattr(p, "pid", None)
                ports_info.append(f"{dev} - {desc} (VID: {vid}, PID: {pid})")

            avail = "\n".join(f"  - {pi}" for pi in ports_info) if ports_info else "  (no serial ports found)"
            raise RuntimeError(f"Could not find serial port for EmagerPuck. Available ports:\n{avail}")

        self.ser = serial.Serial(com_port, baud_rate, timeout=0)
        self.ser.close()

        self._buf = bytearray()
        self.pos = 0

        self.frames_ok = 0
        self.bad_tlr = 0
        self.resyncs = 0
        self.last_ctr = None
        self.ctr_miss = 0

        self.frame_handlers = []
        self._hdr = bytes([self.HDR0, self.HDR1])

        if self.debug:
            print(f"[EmagerPuck] Selected port: {com_port}")

    def connect(self):
        self.ser.open()

        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def clear_buffer(self):
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def add_frame_handler(self, closure: Callable[[int, np.ndarray], None]):
        """
        closure(frame_id:int, emg_block:(136,32) int16)
        """
        self.frame_handlers.append(closure)

    def _emit_frame(self, frame_id: int, emg_block: np.ndarray):
        for h in self.frame_handlers:
            h(int(frame_id), emg_block)

    def _unpack_15bit_msbfirst_signed(
        self,
        packed: bytes,
        n_values: int,
    ) -> np.ndarray:
        """
        Unpack MSB-first packed 15-bit two's complement samples into int16.

        If restore_lsb0 is True:
          output = sign_extended_15bit_value << 1
        This matches the standalone plotting code.
        """

        b = np.frombuffer(packed, dtype=np.uint8)

        acc = np.uint32(0)
        acc_bits = 0
        out = np.empty(n_values, dtype=np.int16)

        bi = 0
        oi = 0
        total_bytes = b.size

        while oi < n_values:
            while acc_bits < 15:
                if bi >= total_bytes:
                    return out[:oi]

                acc = (acc << 8) | np.uint32(b[bi])
                bi += 1
                acc_bits += 8

            acc_bits -= 15
            v15 = np.uint16((acc >> acc_bits) & 0x7FFF)

            # sign extend 15 -> 16
            v16 = np.uint16(v15)
            if v15 & 0x4000:
                v16 = np.uint16(v16 | 0x8000)

            s = np.int16(v16)

            if self.restore_lsb0:
                s = np.int16(s << 1)

            out[oi] = s
            oi += 1

        return out

    def get_data(self) -> bool:
        """
        Reads available serial data, parses complete frames, and emits decoded EMG blocks.

        Returns True if at least one complete frame was parsed and emitted.
        """

        try:
            n_av = self.ser.in_waiting
        except Exception:
            return False

        if n_av <= 0:
            return False

        data = self.ser.read(n_av)
        if not data:
            return False

        self._buf += data
        emitted_any = False

        while True:
            h = self._buf.find(self._hdr, self.pos)

            if h < 0:
                keep = min(len(self._buf), self.FRAME_SIZE - 1)
                self._buf = self._buf[-keep:] if keep else bytearray()
                self.pos = 0
                return emitted_any

            if len(self._buf) - h < self.FRAME_SIZE:
                if h > 0:
                    self._buf = self._buf[h:]
                    self.pos = 0
                else:
                    self.pos = h
                return emitted_any

            t0 = h + self.TRAILER0_I
            t1 = h + self.TRAILER1_I

            if self._buf[t0] == self.TLR0 and self._buf[t1] == self.TLR1:
                self.frames_ok += 1

                c0 = self._buf[h + self.CTR_START + 0]
                c1 = self._buf[h + self.CTR_START + 1]
                c2 = self._buf[h + self.CTR_START + 2]
                c3 = self._buf[h + self.CTR_START + 3]
                frame_id = (c0 << 24) | (c1 << 16) | (c2 << 8) | c3

                if self.last_ctr is not None:
                    expected = (self.last_ctr + 1) & 0xFFFFFFFF
                    if frame_id != expected:
                        self.ctr_miss += 1
                        if self.debug:
                            print(f"[EmagerPuck] Counter miss: got {frame_id}, expected {expected}")

                self.last_ctr = frame_id

                emg_bytes = bytes(self._buf[h + self.EMG_START: h + self.EMG_END])

                emg_vals = self._unpack_15bit_msbfirst_signed(
                    emg_bytes,
                    n_values=self.EMG_VALUES_PER_FRAME,
                )

                if emg_vals.size == self.EMG_VALUES_PER_FRAME:
                    emg_block = emg_vals.reshape(
                        self.SAMPLES_PER_CH_PER_FRAME,
                        self.CHANNELS,
                    )

                    self._emit_frame(frame_id, emg_block)
                    emitted_any = True

                self.pos = h + self.FRAME_SIZE

                if self.pos > (self.FRAME_SIZE * 2):
                    self._buf = self._buf[self.pos:]
                    self.pos = 0

            else:
                self.bad_tlr += 1
                self.resyncs += 1
                self.pos = h + 1

                if self.pos > (self.FRAME_SIZE * 2):
                    self._buf = self._buf[self.pos:]
                    self.pos = 0


# ============================================================
# Streamer process
# ============================================================

class EmagerPuckStreamer(Process):
    def __init__(self, shared_memory_items, emager_kwargs: dict | None = None):
        super().__init__(daemon=True)
        self.shared_memory_items = shared_memory_items
        self._stop_event = Event()
        self.e = None
        self.emager_kwargs = emager_kwargs or {}

        self._shapes = {item[0]: item[1] for item in shared_memory_items if len(item) >= 2}

        self._q = None
        self._writer = None
        self.drop_count = 0

    def run(self):
        self.smm = SharedMemoryManager()

        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        bw = self.emager_kwargs

        baud = int(bw.get("baud_rate", 3000000))
        com_name = bw.get("com_name", None)
        vid_pid = bw.get("vid_pid", (12259, 256))
        restore_lsb0 = bool(bw.get("restore_lsb0", True))
        debug = bool(bw.get("debug", False))

        self.e = EmagerPuck(
            baud_rate=baud,
            com_name=com_name,
            vid_pid=vid_pid,
            restore_lsb0=restore_lsb0,
            debug=debug,
        )

        self.e.connect()
        self.e.clear_buffer()

        self._q = Queue(maxsize=200)

        def buffer_write(tag: str, data: np.ndarray) -> None:
            """
            Prepend data (N,D) to shared memory buffer tag (H,D),
            keeping buffer size fixed, and increment tag_count by N.
            """

            if data is None:
                return

            if data.ndim == 1:
                data = data.reshape(1, -1)

            if data.ndim != 2:
                return

            count_tag = f"{tag}_count"

            def add_to_buffer(buffer, new=data):
                new_buffer = np.vstack((new[::-1], buffer))
                return new_buffer[:buffer.shape[0], :]

            self.smm.modify_variable(tag, add_to_buffer)

            nb_row = data.shape[0]
            self.smm.modify_variable(count_tag, lambda x, r=nb_row: x + r)

        def writer_thread_fn():
            while not self._stop_event.is_set():
                try:
                    frame_id, emg_block = self._q.get(timeout=0.1)
                except Empty:
                    continue

                try:
                    emg = np.asarray(emg_block, dtype=np.int16)
                    buffer_write("emg", emg)

                except Exception as exc:
                    if debug:
                        print(f"[EmagerPuckStreamer] Writer error: {exc}")

                finally:
                    self._q.task_done()

        self._writer = threading.Thread(target=writer_thread_fn, daemon=True)
        self._writer.start()

        def on_frame(frame_id, emg_block):
            try:
                self._q.put_nowait((int(frame_id), emg_block))
            except Exception:
                self.drop_count += 1
                if self.drop_count % 10 == 0:
                    try:
                        print("DROPPED puck frames:", self.drop_count, "qsize:", self._q.qsize())
                    except Exception:
                        print("DROPPED puck frames:", self.drop_count)

        self.e.add_frame_handler(on_frame)

        try:
            while not self._stop_event.is_set():
                did = self.e.get_data()

                if not did:
                    time.sleep(0.001)

        finally:
            self._cleanup()

    def stop(self):
        self._stop_event.set()
        self.join()

    def _cleanup(self):
        try:
            if self.e is not None:
                self.e.close()
        except Exception:
            pass

        try:
            if hasattr(self, "smm") and self.smm is not None:
                self.smm.cleanup()
        except Exception:
            pass