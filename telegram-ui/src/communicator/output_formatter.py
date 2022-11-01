class OutputFormatter:
    def __result(self, value):
        return "~ You have {0:.2f}% chance to have depression ~\n".format(value * 100)

    def __depression_information_menu(self, options):
        menu = '==================================\n' + \
               '‖ Find out more about depression ‖\n' + \
               '==================================\n'
        for i in range(len(options)):
            menu += str(i) + ' - ' + options[i] + '\n'
        menu += 'Please enter the number:'
        return menu

    def __need_help(self, helplines):
        return "============================================\n" + \
               "‖ Please consider the following helplines: ‖\n" + \
               "============================================\n" + \
               helplines

    def opening(self):
        return "Please tell me more about yourself."

    def error(self):
        return "Something went wrong..."

    def evaluation_response(self, response, cur, max):
        return response + "\n" + \
               "[Evaluating - {}/{}]".format(cur, max)

    def chat_response(self, response, value):
        return self.__result(value) + response

    def init_info_response(self, options, value):
        return self.__result(value) + self.__depression_information_menu(options)

    def info_response(self, information, options, value):
        return self.__result(value) + information + '\n\n' + self.__depression_information_menu(options)

    def help_response(self, helplines, value):
        return self.__result(value) + self.__need_help(helplines)

    def invalid_option(self):
        return 'Please provide a valid option.'
