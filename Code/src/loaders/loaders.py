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
    return pd.DataFrame(data_64['train_history']), None
  
def _reshape_progan(data):
    # NOTE: USED AI TO RESHAPE PROGAN DATA INTO A CORRECT DATA STRUCTURE
    th = data['train_history']
    n = len(th['gen_loss'])
    df = pd.DataFrame({
        'epoch': range(1, n + 1),
        'generator_loss': th['gen_loss'],
        'discriminator_loss': th['disc_loss'],
    })
    fid_dict = dict(zip(th['fid_epochs'], th['fid']))
    df['fid'] = df['epoch'].map(fid_dict)
    return df


def load_progan_data():
    with open('logs/progan_64_history.json', 'r') as f:
        data_64 = json.load(f)
    with open('logs/progan_128_history.json', 'r') as f:
        data_128 = json.load(f)
    return _reshape_progan(data_64), _reshape_progan(data_128)


def load_loss_data(model_type):
    if model_type == 'wgan':
        return load_wgan_data()
    elif model_type == 'dcgan':
        return load_dcgan_data()
    elif model_type == 'progan':
        return load_progan_data()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
