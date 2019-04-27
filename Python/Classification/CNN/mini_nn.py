import numpy

class Module:
    pass

class Conv2d:
    def __init__(input_channal,
                 output_channal,
                 kernal_size,
                 bias=True,
                 stride=1,
                 padding=0):
        self.input_channal = input_channal
        self.output_channal = output_channal
        self.kernal_size = kernal_size
        self.bias = bias
        self.stride = stride
        self.padding = padding

    pass


class MaxPool2d:
    pass

class linear:
    pass

