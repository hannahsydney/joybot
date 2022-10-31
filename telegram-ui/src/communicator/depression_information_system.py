import openpyxl

class DepressionInformationSystem:
    def __init__(self, file):
        # Read from excel file
        workbook = openpyxl.load_workbook(file)
        worksheet = workbook.active
        self.depression_information = {}

        # Store depression information as key(title)-value(information) pairs
        for row in range(1, worksheet.max_row + 1):
            key = worksheet['A{}'.format(row)].value
            value = worksheet['B{}'.format(row)].value
            self.depression_information[key] = value

        # Store depression information's titles as options to retrieve information
        self.options = list(self.depression_information.keys())

    def __get_option_information(self, index):
        # Get option
        option = self.options[index]
        # Get and return corresponding information
        return option + ':\n' + self.depression_information[option]

    def get_options(self):
        return self.options

    def get_information(self, option_index_str):
        # Convert option index from string to int
        option_index = int(option_index_str)
        # Validate option index
        if option_index < 0 or option_index > len(self.options):
            raise Exception()
        # Return option information
        return self.__get_option_information(option_index)

    def get_helplines(self):
        # Get and return helplines information
        index = len(self.options) - 1 # helplines is the last option
        return self.__get_option_information(index)
