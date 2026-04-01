import torch
from custom_lstm import FastLSTM

org_model = torch.nn.LSTM(input_size=10, hidden_size=20, num_layers=2, bias=True, bidirectional=False).cuda()
new_model = FastLSTM(input_size=10, hidden_size=20, num_layers=2, bias=True, bidirectional=False, activation_impl="moffett").cuda()
new_model = new_model.to(torch.bfloat16)
new_model = new_model.from_torch_lstm(org_model).cuda().to(torch.bfloat16)


inputs = torch.randn(5, 3, 10).cuda()  # (seq_len, batch_size, input_size)
org_output = org_model(inputs)
new_output = new_model(inputs.to(torch.bfloat16))
print("Original Model Output:", org_output[0])
print("New Model Output:", new_output[0])