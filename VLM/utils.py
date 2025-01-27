from config import SimpleVLMConfig

class Colors:
    RED = '\033[1m\033[31m'
    GREEN = '\033[1m\033[32m'
    BLUE = '\033[1m\033[34m'
    YELLOW = '\033[1m\033[33m'
    MAGENTA = '\033[1m\033[35m'
    CYAN = '\033[1m\033[36m'
    RESET = '\033[1m\033[0m'

def config_check(config: SimpleVLMConfig):
    if not config.image_pad_token_id:
        raise ValueError('image_pad_token_id is not assigned!')