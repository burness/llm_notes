from zxyGPT.model import GPT
from sort_dataset import SortDataset


train_dataset = SortDataset("train")

model_config = GPT.get_default_config()
model_config.model_type = "gpt-nano"
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

from zxyGPT.trainer import Trainer
train_config = Trainer.get_default_config()
train_config.learning_rate =5e-4
train_config.max_iters = 1000
train_config.batch_size = 32
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 ==0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback("on_batch_end", batch_end_callback)
trainer.run()
