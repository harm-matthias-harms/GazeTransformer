from model.transformer import GazeTransformer
import pytorch_lightning as pl

model = GazeTransformer(16, 512, 12) #592)
trainer = pl.Trainer(gpus=-1)

trainer.fit(model)
