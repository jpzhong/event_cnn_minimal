"""
Delta-version of recurrent modules
"""
import torch


class DeltaConvGRU(torch.nn.Module):
    """
    Generate a Delta-convolutional GRU cell
    Adapted from:
    https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size,
                 threshold=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold

        self.reset_gate = torch.nn.Conv2d(input_size + hidden_size,
                                          hidden_size,
                                          kernel_size,
                                          padding=padding)
        self.update_gate = torch.nn.Conv2d(input_size + hidden_size,
                                           hidden_size,
                                           kernel_size,
                                           padding=padding)
        self.out_gate_x = torch.nn.Conv2d(input_size,
                                          hidden_size,
                                          kernel_size,
                                          padding=padding)
        self.out_gate_h = torch.nn.Conv2d(hidden_size,
                                          hidden_size,
                                          kernel_size,
                                          padding=padding)

        torch.nn.init.orthogonal_(self.reset_gate.weight)
        torch.nn.init.orthogonal_(self.update_gate.weight)
        torch.nn.init.orthogonal_(self.out_gate_x.weight)
        torch.nn.init.orthogonal_(self.out_gate_h.weight)
        torch.nn.init.constant_(self.reset_gate.bias, -1.)
        torch.nn.init.constant_(self.update_gate.bias, 0.)
        torch.nn.init.constant_(self.out_gate_x.bias, 0.)
        torch.nn.init.constant_(self.out_gate_h.bias, 0.)

    def forward(self, x, prev_state=None, prev_memory=None):
        if prev_state is None:
            state_size = [x.shape[0], self.hidden_size, *x.shape[2:]]
            prev_state = torch.zeros(state_size).to(x.device)

        memory = {}
        # data size is [batch, channel, height, width]
        if prev_memory is not None:
            # Delta and thresholding
            delta_h = prev_state - prev_memory['prev_prev_state']
            delta_x = x - prev_memory['prev_input']
            if not self.training and self.threshold > 0:
                delta_inp_abs = torch.abs(delta_x)
                delta_x = \
                    delta_x.masked_fill_(delta_inp_abs < self.threshold, 0)
                delta_hid_abs = torch.abs(delta_h)
                delta_h = \
                    delta_x.masked_fill_(delta_hid_abs < self.threshold, 0)

            # Delta Rule
            stacked_inputs = torch.cat([delta_x, delta_h], dim=1)
            memory['u'] = self.update_gate(stacked_inputs) + prev_memory['u']
            memory['r'] = self.reset_gate(stacked_inputs) + prev_memory['r']
            memory['c_x'] = self.out_gate_x(delta_x) + prev_memory['c_x']
            memory['c_h'] = self.out_gate_h(delta_h) + prev_memory['c_h']
        else:
            stacked_inputs = torch.cat([x, prev_state], dim=1)
            memory['u'] = self.update_gate(stacked_inputs)
            memory['r'] = self.reset_gate(stacked_inputs)
            memory['c_x'] = self.out_gate_x(x)
            memory['c_h'] = self.out_gate_h(prev_state)

        out_inputs = \
            torch.tanh(memory['c_x'] + memory['r'].sigmoid() * memory['c_h'])

        new_state = prev_state * (1 - memory['u'].sigmoid()) \
                    + out_inputs * memory['u'].sigmoid()

        memory['prev_input'] = x
        memory['prev_prev_state'] = prev_state

        return new_state, memory

    def update_threshold(self, threshold):
        self.threshold = threshold


class DeltaConvLSTM(torch.nn.Module):
    """
    Adapted from:
    https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(DeltaConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a zero tensor to avoid reallocating memory at inference
        # if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = torch.nn.Conv2d(input_size + hidden_size,
                                     4 * hidden_size,
                                     kernel_size,
                                     padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_prev_state, if None is provided
        if prev_state is None:
            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] +
                               list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a zero tensor with size `spatial_size`
                # (if it has not been allocated already)
                self.zero_tensors[state_size] = (torch.zeros(state_size).to(
                    input_.device), torch.zeros(state_size).to(input_.device))

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell
