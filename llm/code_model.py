from transformers import AdamW, get_linear_schedule_with_warmup
from utils import get_model, ModelType
import pytorch_lightning as pl
import torch
from peft import LoraConfig, LoraModel


class CodeModel(pl.LightningModule):
    def __init__(self,
                 training_dataloader,
                 validating_dataloader,
                 testing_dataloader,
                 model_name,
                 model_type,
                 use_lora=False,
                 lr=5e-5,
                 num_train_epochs=100,
                 warmup_steps=1000):
        super().__init__()
        self.training_dataloader = training_dataloader
        self.validating_dataloader = validating_dataloader
        self.testing_dataloader = testing_dataloader
        self.model = get_model(model_name, model_type)
        if use_lora:
            task_type = model_type
            if model_type != ModelType.CAUSAL_LM:
                task_type = "SEQ_2_SEQ_LM"
            peft_config = LoraConfig(
              task_type=task_type,
              r=8,
              lora_alpha=32,
              inference_mode=False,
              lora_dropout=0.01,
            )
            self.model = LoraModel(self.model, peft_config, "default")
        self.lr = lr
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(self.training_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=self.hparams.warmup_steps,
                                                                     num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return self.training_dataloader

    def val_dataloader(self):
        return self.validating_dataloader

    def test_dataloader(self):
        return self.testing_dataloader
