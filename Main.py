import configparser
from need.models.CustomMain import CustomMain
from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()
    config = configparser.ConfigParser()
    config.read('config.ini')
    Process(target=CustomMain(config)).start()

