import logging
import colorlog

def setup_logger(name='TaawnAlgo', prefix=None):
    """Konfiguriert einen farbigen Logger für TaawnAlgo"""
    
    # Existierenden Logger abrufen oder neu erstellen
    logger = logging.getLogger(name)

    logger.propagate = True
    
    # Wenn der Logger bereits Handler hat, nicht noch einen hinzufügen
    if logger.handlers:
        return logger
    
    # Root Logger deaktivieren
    logging.getLogger().handlers = []
    
    # Handler erstellen
    handler = colorlog.StreamHandler()
    
    # Farbiges Formatter erstellen mit optionalem Prefix
    log_format = "%(log_color)s%(asctime)s - %(name)s"
    if prefix:
        log_format += f"[{prefix}]"
    log_format += " - %(levelname)s - %(message)s"
    
    formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger