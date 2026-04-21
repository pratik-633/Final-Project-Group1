import streamlit as st
import pandas as pd

def plot_fid(train_history):
    fid_df = (
        train_history[['epoch', 'fid']]
        .dropna(subset=['fid'])
        .set_index('epoch')
    )
    st.line_chart(fid_df)

def plot_generator_loss(train_history):
    gen_loss = train_history['generator_loss']
    epochs = train_history['epoch']
    
    gen_loss_df = pd.DataFrame({'epoch': epochs, 'generator_loss': gen_loss}).set_index('epoch')
    st.line_chart(gen_loss_df)
    
def plot_discriminator_loss(train_history, model):
  if model == 'wgan':
    loss = train_history['critic_loss']
  else:
    loss = train_history['discriminator_loss']
  
  epochs = train_history['epoch']
  disc_loss_df = pd.DataFrame({'epoch': epochs, 'discriminator_loss': loss}).set_index('epoch')
  st.line_chart(disc_loss_df)
  
def plot_loss_curves(train_history, metric, model_type):
  if metric == 'generator':
    plot_generator_loss(train_history)
  elif metric == 'discriminator':
    if model_type=='wgan':
      plot_discriminator_loss(train_history, model_type)
    else:
      plot_discriminator_loss(train_history, model_type)
  elif metric == 'fid':
    plot_fid(train_history)
  