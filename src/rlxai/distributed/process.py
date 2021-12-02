# Manage
import traceback
import time


def manage_process(
    Agent,
    agent_config,
    result_queue,
    sync_queue,
    run_step,
    print_period,
):
    agent = Agent(**agent_config)

    step, print_stamp = 0, 0
    try:
        while step < run_step:
            wait = True
            while wait or not result_queue.empty():
                _step, result = result_queue.get()
                wait = False
            print_stamp += _step - step
            step = _step
            if print_stamp >= print_period or step >= run_step:
                agent.sync_in(**sync_queue.get())
                print_stamp = 0
    except Exception as e:
        traceback.print_exc()
