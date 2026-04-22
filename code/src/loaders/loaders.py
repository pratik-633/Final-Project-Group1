import json
import pandas as pd


def load_wgan_data():
    with open('logs/wgan_gp_64_history.json', 'r') as f:
        data_64 = json.load(f)
    with open('logs/wgan_gp_128_history.json', 'r') as f:
        data_128 = json.load(f)
    return pd.DataFrame(data_64['train_history']), pd.DataFrame(data_128['train_history'])
    

def load_dcgan_data():
    with open('logs/dcgan_64_history.json', 'r') as f:
        data_64 = json.load(f)
    with open('logs/dcgan_128_history.json', 'r') as f:
        data_128 = json.load(f)
    return pd.DataFrame(data_64['train_history']), pd.DataFrame(data_128['train_history'])
  
def load_progan_data():
    with open('logs/progan_64_history.json', 'r') as f:
        data_64 = json.load(f)
    with open('logs/progan_128_history.json', 'r') as f:
        data_128 = json.load(f)
    return pd.DataFrame(data_64['train_history']), pd.DataFrame(data_128['train_history'])


def load_loss_data(model_type):
    if model_type == 'wgan':
        return load_wgan_data()
    elif model_type == 'dcgan':
        return load_dcgan_data()
    elif model_type == 'progan':
        return load_progan_data()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
