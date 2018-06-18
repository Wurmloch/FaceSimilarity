import sys


def set_progress(min_num, max_num, progress_num):
    """
    sets the progress to a number between 0 and 1
    :param min_num: the minimum number in the range
    :type min_num: int
    :param max_num: the maximum number in the range
    :type max_num: int
    :param progress_num: the current progress in the range
    :type progress_num: int
    :return:
    """
    z = ((progress_num - min_num) / (max_num - min_num)) * 100
    sys.stdout.write("\r%d%%" % z)
    sys.stdout.flush()
