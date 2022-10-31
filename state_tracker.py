from state import State


class StateTracker:
    def __init__(self, max_eval_inputs, low_threshold, high_threshold):
        self.max_eval_inputs = max_eval_inputs
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.cur_eval_inputs = 0
        self.state = State.EVAL

    def update(self, value):
        # Only update inputs count when EVAL state
        if self.state == State.EVAL:
            self.cur_eval_inputs += 1
            # Retain state if current input count is less than max input count
            if self.cur_eval_inputs < self.max_eval_inputs:
                return
        
        # Set state according to value
        if value < self.low_threshold:      # Low: continue chatting
            self.state = State.CHAT
        elif value < self.high_threshold:   # Medium: share depression information
            self.state = State.INFO
        else:                               # High: display helplines
            self.state = State.HELP

    def get_state(self):
        return self.state

    def get_cur_eval_inputs(self):
        return self.cur_eval_inputs

    def get_max_eval_inputs(self):
        return self.max_eval_inputs
