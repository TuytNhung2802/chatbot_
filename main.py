from omegaconf import OmegaConf
from model import ChatBot


def main():
    config  = OmegaConf.load("./config.yaml")

    chatbot = ChatBot(config)
    chatbot.Trainer(is_retrain=False)


if __name__=='__main__':
    main()