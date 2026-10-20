"""Gate for the #3-on-Sunday-morning session loop.

Prints READY once local time reaches the target (Sun 2026-06-14 08:00 local),
else WAIT <seconds-remaining>. The loop uses this so a spurious/early wake does
not dispatch the EXISTENT_WRONG_ID sub-study before its scheduled time.
"""
import datetime
import time

TARGET = datetime.datetime(2026, 6, 14, 8, 0, 0)  # Sunday 08:00 America/Chicago (local)


def main() -> None:
    remaining = int(TARGET.timestamp() - time.time())
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %A")
    if remaining <= 0:
        print(f"READY (now {now}; target {TARGET:%Y-%m-%d %H:%M %A})")
    else:
        print(f"WAIT {remaining} (now {now}; ~{remaining/3600:.1f}h to "
              f"{TARGET:%Y-%m-%d %H:%M %A})")


if __name__ == "__main__":
    main()
