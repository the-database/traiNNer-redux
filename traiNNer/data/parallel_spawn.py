"""Start Windows DataLoader workers in parallel instead of one after another.

On Windows a worker is a new process. The parent writes the pickled worker (torch's
worker function and the dataset) into a pipe, and the worker unpickles it while it
reads, importing torch and traiNNer on the way. The write blocks once the pipe is
full, so the parent waits through those imports, about 5 s per worker, before it can
start the next one: 8 workers took 45 s to start. Here the pickle is built up front as
usual, but written to the pipe from a background thread, so the workers import in
parallel.
"""

import io
import logging
import multiprocessing.context as mp_context
import sys
import threading
from multiprocessing import reduction, util
from typing import Any

logger = logging.getLogger(__name__)

# private names of the stdlib's Windows spawn code that ParallelSpawnPopen reuses
_NEEDED = (
    "WINENV",
    "_path_eq",
    "_close_handles",
    "STARTUPINFO",
    "STARTF_FORCEOFFFEEDBACK",
    "_winapi",
)
_PSW: dict[str, Any] | None = None

if sys.platform == "win32":
    import msvcrt
    import os
    from multiprocessing import popen_spawn_win32, spawn

    if all(hasattr(popen_spawn_win32, name) for name in _NEEDED):
        _PSW = {name: getattr(popen_spawn_win32, name) for name in _NEEDED}

    class ParallelSpawnPopen(popen_spawn_win32.Popen):
        """popen_spawn_win32.Popen, except that the data for the child is sent from
        a thread."""

        def __init__(self, process_obj: Any) -> None:
            assert _PSW is not None
            winapi = _PSW["_winapi"]
            prep_data = spawn.get_preparation_data(process_obj._name)  # noqa: SLF001
            rhandle, whandle = winapi.CreatePipe(None, 0)
            wfd = msvcrt.open_osfhandle(whandle, 0)
            cmd = spawn.get_command_line(parent_pid=os.getpid(), pipe_handle=rhandle)
            python_exe = spawn.get_executable()
            if _PSW["WINENV"] and _PSW["_path_eq"](python_exe, sys.executable):
                cmd[0] = python_exe = getattr(sys, "_base_executable")  # noqa: B009
                env = os.environ.copy()
                env["__PYVENV_LAUNCHER__"] = sys.executable
            else:
                env = None
            cmd_line = " ".join(f'"{x}"' for x in cmd)

            to_child = open(wfd, "wb", closefd=True)  # noqa: SIM115
            try:
                hp, ht, pid, _tid = winapi.CreateProcess(
                    python_exe,
                    cmd_line,
                    None,
                    None,
                    False,
                    0,
                    env,
                    None,
                    _PSW["STARTUPINFO"](dwFlags=_PSW["STARTF_FORCEOFFFEEDBACK"]),
                )
                winapi.CloseHandle(ht)
            except BaseException:
                winapi.CloseHandle(rhandle)
                to_child.close()
                raise

            self.pid = pid
            self.returncode = None
            self._handle = hp
            self.sentinel = int(hp)
            self.finalizer = util.Finalize(
                self, _PSW["_close_handles"], (self.sentinel, int(rhandle))
            )

            # Pickling must happen here: handles are duplicated into the child
            # while this Popen is the one spawning.
            buf = io.BytesIO()
            mp_context.set_spawning_popen(self)
            try:
                reduction.dump(prep_data, buf)
                reduction.dump(process_obj, buf)
            finally:
                mp_context.set_spawning_popen(None)
            data = buf.getvalue()

            def send() -> None:
                try:
                    with to_child:
                        to_child.write(data)
                except OSError as e:  # the child exited before reading
                    logger.debug("Worker %d did not read its data: %s", pid, e)

            threading.Thread(target=send, name=f"spawn-{pid}", daemon=True).start()

    class ParallelSpawnProcess(mp_context.SpawnProcess):
        @staticmethod
        def _Popen(process_obj: Any) -> ParallelSpawnPopen:  # noqa: N802
            return ParallelSpawnPopen(process_obj)

    class ParallelSpawnContext(mp_context.SpawnContext):
        Process = ParallelSpawnProcess  # pyright: ignore[reportAssignmentType]


def parallel_spawn_context() -> mp_context.BaseContext | None:
    """A multiprocessing context that starts workers in parallel on Windows, or None
    where the default is fine (Linux forks) or this Python's internals don't match."""
    if sys.platform != "win32" or _PSW is None:
        return None
    return ParallelSpawnContext()
