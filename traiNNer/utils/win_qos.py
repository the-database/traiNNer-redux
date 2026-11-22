"""Keep Windows from treating training processes as low-priority background work.

Windows 11 lowers the quality of service of processes it considers background work,
for example ones with no foreground window, and on CPUs with performance and
efficiency cores it then runs their threads on the efficiency cores only. traiNNer's
DataLoader workers have no window, so with 16 workers on an i9-12900K all of them
shared the 8 efficiency cores while the performance cores sat idle. Opting out of
execution speed throttling lets the scheduler use every core again.
"""

import sys


def disable_power_throttling() -> bool:
    """Opt the current process out of Windows execution speed throttling (EcoQoS).

    Returns True when the setting was applied. Does nothing on other systems.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class ProcessPowerThrottlingState(ctypes.Structure):
        _fields_ = (
            ("Version", wintypes.ULONG),
            ("ControlMask", wintypes.ULONG),
            ("StateMask", wintypes.ULONG),
        )

    process_power_throttling = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
    execution_speed = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    # Taking control of execution speed with its state bit clear turns throttling off.
    state = ProcessPowerThrottlingState(1, execution_speed, 0)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    return bool(
        kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            process_power_throttling,
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
    )
