

def get_step_scheduler(start, alpha, every_perc):
    def func(progress_remaining):
        progress_done = 1 - progress_remaining
        num_steps = (progress_done//every_perc)
        lr = start * (alpha**num_steps)

        return lr
    
    return func

def get_exp_scheduler(start, end):
    def func(progress_remaining):
        progress_done = 1 - progress_remaining
        lr = start * ((end/start)**(progress_done))

        return lr
    
    return func