# Run this in your notebook AFTER training, to save what the API needs

import json
import torch

torch.save(model.state_dict(), "model.pth")

with open("vocab.json", "w") as f:
    json.dump(vocab, f)

print("Saved model.pth and vocab.json")