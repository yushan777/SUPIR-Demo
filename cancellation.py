"""
Shared cancellation flag for graceful processing interruption
"""
from threading import Event

# Global cancellation flag
CANCEL_FLAG = Event()

def cancel_processing():
    """Set the cancellation flag"""
    CANCEL_FLAG.set()
    print("⚠️ Cancellation requested by user")

def reset_cancel_flag():
    """Clear the cancellation flag for the next run"""
    CANCEL_FLAG.clear()

def check_cancelled():
    """Check if cancellation has been requested"""
    return CANCEL_FLAG.is_set()
