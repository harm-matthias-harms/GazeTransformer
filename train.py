from model.transformer import GazeTransformer
import pytorch_lightning as pl

model = GazeTransformer()
trainer = pl.Trainer(gpus=-1)

trainer.fit(model)
