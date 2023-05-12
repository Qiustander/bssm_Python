from time import process_time

def get_runtime(loop_time=10):
    def outer_wrapper(func):
        def timer(*args, **kwargs):
            total_time = 0
            for _ in range(loop_time):
                start = process_time()
                result = func(*args, **kwargs)
                end = process_time()

                total_time += (end - start)
            print(f"average execute time is {total_time / loop_time}")

        return timer

    return outer_wrapper