from datetime import datetime


def get_strftime_now(fmt='%Y%m%d_%H%M'):
    return datetime.now().strftime(fmt)