import os
from communicator.depression_information_system import DepressionInformationSystem
from communicator.output_formatter import OutputFormatter
from communicator.state_tracker import StateTracker
from communicator.state import State
from godel.godel import generate
from depressionDetector.depressionDetector import Detector

from dotenv import load_dotenv
load_dotenv()

INFORMATION_PATH = os.environ['depression_information_path']

class Communicator:
    def __init__(self, max_eval_inputs=2, low_threshold=0.3, high_threshold=0.7, depression_information_file=INFORMATION_PATH):
        self.state_tracker = StateTracker(
            max_eval_inputs, low_threshold, high_threshold)
        self.dep_info_sys = DepressionInformationSystem(
            depression_information_file)
        self.output_formatter = OutputFormatter()
        self.depression_detector = Detector()

    def start(self):
        return self.output_formatter.opening()

    def __eval(self, user_input):
        state = self.state_tracker.get_state()
        # Do evaluation for only EVAL or CHAT
        if state == State.EVAL or state == state.CHAT:
            self.depression_detector.depressionDetection(user_input)
            # TODO: Pass input to depression detector
            # TODO: Pass result to depression scaler
            self.state_tracker.update(self.depression_detector.getScore())

    def __get_state_eval_response(self, history: list):
        # Interact with GODEL and return response with evaluation status
        cur = self.state_tracker.get_cur_eval_inputs()
        max = self.state_tracker.get_max_eval_inputs()
        godel_response = generate(history)
        return self.output_formatter.evaluation_response(godel_response, cur, max)

    def __get_state_chat_response(self, value, history: list):
        # Interact with GODEL and return response with result
        godel_response = generate(history)
        return self.output_formatter.chat_response(godel_response, value)

    def __get_state_info_response(self, value, user_input, prev_state):
        # Get options
        options = self.dep_info_sys.get_options()
        if prev_state != self.state_tracker.get_state():
            # Return result and depression information menu upon first enter INFO state
            return self.output_formatter.init_info_response(options, value)
        else:
            # Return result, depression information and menu
            try:
                information = self.dep_info_sys.get_information(user_input)
                return self.output_formatter.info_response(information, options, value)
            except:
                return self.output_formatter.invalid_option()

    def __get_state_help_response(self, value):
        # Get helplines and return with result
        helplines = self.dep_info_sys.get_helplines()
        return self.output_formatter.help_response(helplines, value)

    def __get_state_response(self, user_input, prev_state, history: list):
        state = self.state_tracker.get_state()
        value = self.depression_detector.getScore()

        if state == State.EVAL:
            return self.__get_state_eval_response(history)
        elif state == State.CHAT:
            return self.__get_state_chat_response(value, history)
        elif state == State.INFO:
            return self.__get_state_info_response(value, user_input, prev_state)
        elif state == State.HELP:
            return self.__get_state_help_response(value)
        else:
            return self.output_formatter.error()

    def handle_input(self, user_input, history: list):
        prev_state = self.state_tracker.get_state()
        self.__eval(user_input)
        return self.__get_state_response(user_input, prev_state, history)


def init():
    return Communicator()

