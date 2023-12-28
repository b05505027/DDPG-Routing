class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
    
    def write(self, *message):
        message = ' '.join(map(str, message))
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')