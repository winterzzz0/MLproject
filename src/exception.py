import sys
import logging
import src.logger

def error_message_details(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() #traceback
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "ERROR occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)
    
    def __str__(self):
        return self.error_message


if __name__ == "__main__": #making sure it works
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divided by zero")
        raise CustomException(e,sys)