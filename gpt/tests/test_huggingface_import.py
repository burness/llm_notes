import unittest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from zxyGPT.model import GPT
from zxyGPT.bpe import BPETokenizer

class TestHuggingFaceImport(unittest.TestCase):

    def test_gpt2(self):
        model_type = "gpt2"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prompt = "Hello!!!!!!!!!? I love zxy and xnn"
 
        model = GPT.from_pretrained(model_type)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) # init a HF model too

        model.to(device)
        model_hf.to(device)

        model.eval()
        model_hf.eval()

        tokenizer = BPETokenizer()
        x1 = tokenizer(prompt).to(device)
        tokenizer_hf = GPT2Tokenizer.from_pretrained(model_type)
        model_hf.config.pad_token_id = model_hf.config.eos_token_id # suppress a warning
        encoded_input = tokenizer_hf(prompt, return_tensors='pt').to(device)
        x2 = encoded_input['input_ids']
        print("x1 token ids: {0}, x2 token ids".format(x1, x2))

        # ensure the logits match exactly
        logits1, loss = model(x1)
        logits2 = model_hf(x2).logits
        self.assertTrue(torch.allclose(logits1, logits2))

        # now draw the argmax samples from each
        y1 = model.generate(x1, max_new_tokens=20, do_sample=False)[0]
        y2 = model_hf.generate(x2, max_new_tokens=20, do_sample=False)[0]
        self.assertTrue(torch.equal(y1, y2)) # compare the raw sampled indices

        # convert indices to strings
        out1 = tokenizer.decode(y1.cpu().squeeze())
        out2 = tokenizer_hf.decode(y2.cpu().squeeze())
        self.assertTrue(out1 == out2) # compare the exact output strings too


if __name__ == "__main__":
    unittest.main()