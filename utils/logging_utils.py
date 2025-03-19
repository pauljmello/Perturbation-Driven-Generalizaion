import sys
import logging

class Colors:
    RESET = "\033[0m"

    # colors
    NEON_PINK    = "\033[38;5;198m"
    NEON_CYAN    = "\033[38;5;51m"
    NEON_GREEN   = "\033[38;5;46m"
    NEON_YELLOW  = "\033[38;5;226m"
    NEON_ORANGE  = "\033[38;5;208m"
    NEON_PURPLE  = "\033[38;5;135m"
    NEON_BLUE    = "\033[38;5;33m"
    NEON_RED     = "\033[38;5;196m"
    NEON_LIME    = "\033[38;5;118m"
    NEON_MAGENTA = "\033[38;5;201m"

    # Dark neon
    DARK_ORANGE  = "\033[38;5;166m"
    DARK_MAGENTA = "\033[38;5;125m"

class Formatter(logging.Formatter):

    LEVEL_COLORS = {
        logging.DEBUG:    Colors.NEON_BLUE,
        logging.INFO:     Colors.NEON_GREEN,
        logging.WARNING:  Colors.NEON_YELLOW,
        logging.ERROR:    Colors.NEON_RED,
        logging.CRITICAL: Colors.NEON_MAGENTA,
    }

    COMPONENT_COLORS = {
        'main':          Colors.NEON_PINK,
        'trainer':       Colors.NEON_CYAN,
        'models':        Colors.NEON_PURPLE,
        'visualization': Colors.NEON_BLUE,
        'config':        Colors.NEON_LIME,
        'augmentation':  Colors.NEON_YELLOW,
        'validators':    Colors.DARK_MAGENTA,
        'report':        Colors.NEON_MAGENTA,
        'checkpoint':    Colors.DARK_ORANGE,
        'metrics':       Colors.NEON_GREEN,
        'callbacks':     Colors.NEON_YELLOW,
    }

    def format(self, record):
        orig_level, orig_name = record.levelname, record.name
        record.levelname = f"{self.LEVEL_COLORS.get(record.levelno, '')}{record.levelname}{Colors.RESET}"
        record.name = next((f"{clr}{record.name}{Colors.RESET}"
                            for comp, clr in self.COMPONENT_COLORS.items()
                            if comp in record.name), record.name)
        msg = super().format(record)
        record.levelname, record.name = orig_level, orig_name
        return msg

def setup_logging(level=logging.INFO):
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = [handler]
    return logger
