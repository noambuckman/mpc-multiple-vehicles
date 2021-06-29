import string, datetime, random


def random_date_string(date_format: str = "%Y%m%d-%H%M%S"):
    '''Generate a log name that has a random string prefix and then a date'''

    alpha_num = string.ascii_lowercase[:8] + string.digits
    experiment_string = ("".join(random.choice(alpha_num)
                                 for j in range(4)) + "-" + "".join(random.choice(alpha_num) for j in range(4)) + "-" +
                         datetime.datetime.now().strftime(date_format))

    return experiment_string