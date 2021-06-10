from model.fixationnetpl import FixationNetPL
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    verbose=True
)

model_checkpoint_callback = ModelCheckpoint(
    dirpath='./model/checkpoints/fixationnet',
    filename='{epoch}-{val_loss:.2f}',
    monitor='val_loss',
    mode='min',
    verbose=True,
    save_top_k=1,
)

model = FixationNetPL(batch_size=512, num_worker=12)
trainer = pl.Trainer(gpus=-1)#, callbacks=[early_stopping_callback, model_checkpoint_callback])

trainer.fit(model)

# best_model = model.load_from_checkpoint(
#     model_checkpoint_callback.best_model_path)

# trainer.test(model)


# possible improvements, when we need to run with small batchsize
# trainer = Trainer(accumulate_grad_batches=1)
# trainer = Trainer(auto_scale_batch_size=None|'power'|'binsearch'); trainer.tune(model)
# trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')