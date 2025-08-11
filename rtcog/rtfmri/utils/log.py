import logging
        
log = logging.getLogger("GENERAL")

def get_logger():
    """
    Configure and return logger with both console and file handlers.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    if not log.handlers:
        log_fmt = logging.Formatter('[%(levelname)s - %(filename)s]: %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_fmt)
        log.addHandler(stream_handler)

        file_handler = logging.FileHandler('main.log', mode='w')
        file_handler.setFormatter(log_fmt)
        log.addHandler(file_handler)

    return log

def set_logger(debug=False, silent=False):
    """
    Set the logging level.

    Parameters
    ----------
    debug : bool
        Whether to set the logging level to DEBUG.
    silent : bool
        Whether to silence all but CRITICAL messages.
    """
    if debug:
        log.setLevel(logging.DEBUG)
    elif silent:
        log.setLevel(logging.CRITICAL)
    else:
        log.setLevel(logging.INFO)
    return log
