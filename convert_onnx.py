"""Convert the public GPT-2 small ONNX export (onnx/models, gpt2-10.onnx) into a PyTorch
state dict for transformers.GPT2Model and run a sanity check."""
import onnx, torch
from onnx import numpy_helper
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

m = onnx.load('models/gpt2-10.onnx')
sd = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in m.graph.initializer if not i.name.isdigit()}
model = GPT2Model(GPT2Config())
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing, missing                      # only the causal-mask buffers are unexpected
torch.save(model.state_dict(), 'models/gpt2_small.pt')
tok = GPT2Tokenizer('models/encoder.json', 'models/vocab.bpe')
ids = torch.tensor([tok.encode('The capital of France is the city of')])
with torch.no_grad():
    h = model(ids).last_hidden_state
print([tok.decode([i]) for i in (h[0, -1] @ model.wte.weight.T).topk(5).indices])
