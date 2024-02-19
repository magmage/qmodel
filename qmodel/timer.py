from timeit import default_timer
# timer decorator
def tictoc(func):
    def wrapper(*args, **kwargs):
        start = default_timer()
        re = func(*args, **kwargs)
        end = default_timer()
        print("*tic*toc* {0}() time lapse: {1:.6f} s".format(func.__qualname__, end - start))
        return re
    return wrapper