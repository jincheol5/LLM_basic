from transformers import Trainer,TrainingArguments,DataCollatorForLanguageModeling
from peft import LoraConfig,get_peft_model

class ModelTrainer:
    @staticmethod
    def fine_tuning(model,dataset):
        """
        """
        
