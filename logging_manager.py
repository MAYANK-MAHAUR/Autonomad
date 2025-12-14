"""
Memory-Only Logging Manager
Optimized for Railway deployment - no file I/O
"""
import sys
import logging
from collections import deque
from datetime import datetime
from typing import Optional


class MemoryLogHandler(logging.Handler):
    """
    Custom handler that stores logs in memory (deque)
    Perfect for cloud deployment where file systems are ephemeral
    """
    
    def __init__(self, maxlen: int = 1000):
        super().__init__()
        self.log_buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": msg,
                "raw": record
            })
        except Exception:
            self.handleError(record)
    
    def get_logs(self, last_n: Optional[int] = None) -> list:
        """Get last N logs (or all if None)"""
        if last_n is None:
            return list(self.log_buffer)
        return list(self.log_buffer)[-last_n:]
    
    def clear(self):
        """Clear log buffer"""
        self.log_buffer.clear()
    
    def get_stats(self) -> dict:
        """Get logging statistics"""
        level_counts = {}
        for log in self.log_buffer:
            level = log["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            "total_logs": len(self.log_buffer),
            "max_capacity": self.maxlen,
            "level_counts": level_counts,
            "oldest_log": self.log_buffer[0]["timestamp"] if self.log_buffer else None,
            "newest_log": self.log_buffer[-1]["timestamp"] if self.log_buffer else None
        }


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class LogManager:
    """
    Centralized logging manager for the trading agent
    Memory-only, no file I/O
    """
    
    _instance = None
    _memory_handler: Optional[MemoryLogHandler] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.loggers = {}
    
    def setup_logger(
        self, 
        name: str, 
        level: str = "INFO",
        max_memory_logs: int = 1000
    ) -> logging.Logger:
        """
        Setup a logger with memory-only storage
        
        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_memory_logs: Maximum number of logs to keep in memory
        
        Returns:
            Configured logger instance
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()
        logger.propagate = False
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(console_handler)
        
        # Memory handler (shared across all loggers)
        if self._memory_handler is None:
            self._memory_handler = MemoryLogHandler(maxlen=max_memory_logs)
            self._memory_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
        
        logger.addHandler(self._memory_handler)
        
        self.loggers[name] = logger
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get existing logger or create new one"""
        if name not in self.loggers:
            return self.setup_logger(name)
        return self.loggers[name]
    
    def get_logs(self, last_n: Optional[int] = None) -> list:
        """Get logs from memory buffer"""
        if self._memory_handler:
            return self._memory_handler.get_logs(last_n)
        return []
    
    def get_stats(self) -> dict:
        """Get logging statistics"""
        if self._memory_handler:
            return self._memory_handler.get_stats()
        return {}
    
    def clear_logs(self):
        """Clear all logs from memory"""
        if self._memory_handler:
            self._memory_handler.clear()
    
    def export_logs_json(self) -> list:
        """Export logs as JSON-serializable list"""
        logs = self.get_logs()
        return [
            {
                "timestamp": log["timestamp"],
                "level": log["level"],
                "message": log["message"]
            }
            for log in logs
        ]


# Global log manager instance
log_manager = LogManager()


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger"""
    return log_manager.get_logger(name)