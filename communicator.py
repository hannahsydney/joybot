from depression_information_system import DepressionInformationSystem
from output import Output
from state_tracker import StateTracker
from state import State


class Communicator:
    def __init__(self, max_eval_inputs=2, low_threshold=0.3, high_threshold=0.7, depression_information_file='data/depression_information.xlsx'):
        self.state_tracker = StateTracker(
            max_eval_inputs, low_threshold, high_threshold)
        self.dep_info_sys = DepressionInformationSystem(
            depression_information_file)
        self.output = Output()

    def start(self):
        return self.output.opening()

    def __eval(self, user_input):
        state = self.state_tracker.get_state()
        # Do evaluation for only EVAL or CHAT
        if state == State.EVAL or state == state.CHAT:
            # TODO: Pass input to depression detector
            # TODO: Pass result to depression scaler
            self.state_tracker.update(0.3)

    def __get_state_eval_response(self):
        # Get current and max evaluation inputs
        cur = self.state_tracker.get_cur_eval_inputs()
        max = self.state_tracker.get_max_eval_inputs()
        # Interact with CakeChat and return response with evaluation status
        # TODO: Pass input to cakechat and return response
        return self.output.evaluation_response("CakeChat response.", cur, max)

    def __get_state_chat_response(self, value):
        # Interact with CakeChat and return response with result
        # TODO: Pass input to cakechat and append response
        return self.output.chat_response("CakeChat response.", value)

    def __get_state_info_response(self, value, user_input, prev_state):
        # Get options
        options = self.dep_info_sys.get_options()
        if prev_state != self.state_tracker.get_state():
            # Return result and depression information menu upon first enter INFO state
            return self.output.init_info_response(options, value)
        else:
            # Return result, depression information and menu
            try:
                information = self.dep_info_sys.get_information(user_input)
                return self.output.info_response(information, options, value)
            except:
                return self.output.invalid_option()

    def __get_state_help_response(self, value):
        # Get helplines and return with result
        helplines = self.dep_info_sys.get_helplines()
        return self.output.help_response(helplines, value)

    def __get_state_response(self, user_input, prev_state):
        state = self.state_tracker.get_state()
        value = 0.3

        if state == State.EVAL:
            return self.__get_state_eval_response()
        elif state == State.CHAT:
            return self.__get_state_chat_response(value)
        elif state == State.INFO:
            return self.__get_state_info_response(value, user_input, prev_state)
        elif state == State.HELP:
            return self.__get_state_help_response(value)
        else:
            return self.output.error()

    def handle_input(self, user_input):
        prev_state = self.state_tracker.get_state()
        self.__eval(user_input)
        return self.__get_state_response(user_input, prev_state)


def main():
    communicator = Communicator()
    exit_line = '\n[Enter bye to exit]'
    print(communicator.start() + exit_line)
    user_input = input()
    while user_input.lower() != 'bye':
        print()
        print(communicator.handle_input(user_input) + exit_line)
        user_input = input()
    print('See you.')
    print()


if __name__ == '__main__':
    main()
