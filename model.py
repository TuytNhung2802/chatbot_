import torch
import os
import pytorch_lightning as pl
import pandas as pd

from transformers import T5ForConditionalGeneration
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import  ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

from dataloader import ChatBotDataModel


class ChatBotModel(pl.LightningModule):
    def __init__(self, device, model_name, lr):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.lr = lr

    def __get_data(self, batch):
        return {'input_ids': batch["input_ids"],
                'attention_mask': batch["attention_mask"],
                'labels': batch["labels"],
                'decoder_attention_mask': batch["decoder_attention_mask"],
        }


    def __get_output(self, batch):
        input_data = self.__get_data(batch)
        outputs = self.model(**input_data)
        return outputs


    def training_step(self, batch, batch_idx):
        outputs = self.__get_output(batch)
        self.log("train_loss", outputs.loss, prog_bar=True, logger=True)
        return outputs.loss


    def validation_step(self, batch, batch_idx):
        outputs = self.__get_output(batch)
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_epoch=True)
        return outputs.loss


    def test_step(self, batch, batch_idx):
        outputs = self.__get_output(batch)
        self.log("test_loss",outputs.loss, prog_bar=True, logger=True)
        return outputs.loss


    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)


class ChatBot():
    def __init__(self, config):
        self.config = config
        if config.common.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self.model_name = config.model.model_name
        self.max_length = config.model.max_length
        self.data_path = config.data.data_path
        self.path_checkpoint = config.common.path_checkpoint
        self.name_checkpoint = config.common.name_checkpoint
        self.path_logger = config.common.path_logger


    def Trainer(self, is_retrain=False):
        data = pd.read_excel(self.data_path, sheet_name=self.config.data.tab_name)

        logger = TensorBoardLogger(self.path_logger, name="chatbot_vit5")

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.path_checkpoint,
            filename=self.name_checkpoint,
            save_top_k=self.config.train.top_k,
            verbose = True,
            monitor="val_loss",
            mode= "min",
            save_weights_only=True
            )
        
        if is_retrain:
            checkpoint_path = os.path.join(self.path_checkpoint, self.name_checkpoint + ".ckpt")
            model = ChatBotModel.load_from_checkpoint(
                                                    checkpoint_path,
                                                    device=self.device,
                                                    model_name=self.model_name,
                                                    lr=self.config.train.learning_rate,
                                                )
        else:
            model = ChatBotModel(
                                self.device,
                                self.model_name,
                                self.config.train.learning_rate,
                            )

        for i in range(self.config.train.loops):
            trainer = pl.Trainer(
                logger=logger,
                callbacks=checkpoint_callback,
                max_epochs=self.config.train.epochs,
            )

            if i > 0:
                checkpoint_path = os.path.join(self.path_checkpoint, self.name_checkpoint + ".ckpt")
                while not os.path.exists(checkpoint_path):
                    continue
                model = ChatBotModel.load_from_checkpoint(checkpoint_path, device=self.device, model_name=self.model_name)

            train, test = train_test_split(data, test_size=self.config.train.train_test_split)
            data_model = ChatBotDataModel(self.device, self.model_name, train, test, self.config.train.batch_size, self.max_length)
            data_model.setup()

            trainer.fit(model=model, datamodule=data_model)
            trainer.test(model=model, datamodule=data_model)
            torch.cuda.empty_cache()