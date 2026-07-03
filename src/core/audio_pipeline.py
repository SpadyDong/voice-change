import threading
import queue
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioPipeline:
    def __init__(
        self,
        process_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        sample_rate: int = 48000,
        block_size: int = 1024,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        self.process_callback = process_callback

        self._stream: Optional[sd.Stream] = None
        self._running = False
        self._input_queue: queue.Queue = queue.Queue(maxsize=10)
        self._output_queue: queue.Queue = queue.Queue(maxsize=10)
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"Audio callback status: {status}")

        # Flatten to mono float32
        audio_in = indata[:, 0].astype(np.float32) if indata.ndim > 1 else indata.astype(np.float32)

        # Try to put into input queue (drop if full to maintain real-time)
        try:
            self._input_queue.put_nowait(audio_in.copy())
        except queue.Full:
            pass

        # Try to get processed audio from output queue
        try:
            audio_out = self._output_queue.get_nowait()
            # Ensure correct length
            if len(audio_out) < len(outdata):
                padded = np.zeros(len(outdata), dtype=np.float32)
                padded[: len(audio_out)] = audio_out
                audio_out = padded
            elif len(audio_out) > len(outdata):
                audio_out = audio_out[: len(outdata)]
        except queue.Empty:
            audio_out = np.zeros(len(outdata), dtype=np.float32)

        # Write to output (stereo or mono depending on device)
        if outdata.ndim > 1:
            outdata[:, 0] = audio_out
            if outdata.shape[1] > 1:
                outdata[:, 1] = audio_out
        else:
            outdata[:] = audio_out.reshape(-1, 1) if outdata.ndim > 1 else audio_out

    def _processing_loop(self) -> None:
        while self._running:
            try:
                audio_in = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.process_callback is not None:
                try:
                    audio_out = self.process_callback(audio_in)
                except Exception as e:
                    print(f"Processing error: {e}")
                    audio_out = audio_in
            else:
                audio_out = audio_in

            try:
                self._output_queue.put_nowait(audio_out)
            except queue.Full:
                pass

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker_thread.start()

        try:
            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype=np.float32,
                channels=1,
                callback=self._audio_callback,
                input_device=self.input_device,
                output_device=self.output_device,
            )
            self._stream.start()
        except Exception as e:
            self._running = False
            raise RuntimeError(f"Failed to start audio stream: {e}")

    def stop(self) -> None:
        with self._lock:
            self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        # Clear queues
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

    def is_running(self) -> bool:
        return self._running

    def set_process_callback(self, callback: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
        self.process_callback = callback

    @staticmethod
    def query_devices() -> list:
        try:
            return list(sd.query_devices())
        except Exception:
            return []
