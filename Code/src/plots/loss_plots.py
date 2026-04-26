import streamlit as st
import pandas as pd

def plot_fid(train_history):
    fid_df = (
        train_history[['fid']]
        .dropna(subset=['fid'])
        .reset_index(drop=True)
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
  
def plot_combined_loss(train_history, model):
    epochs = train_history['epoch']
    gen_loss = train_history['generator_loss']
    disc_loss = train_history['critic_loss'] if model == 'wgan' else train_history['discriminator_loss']
    
    df = pd.DataFrame({
        'epoch': epochs,
        'generator_loss': gen_loss,
        'critic_loss' if model == 'wgan' else 'discriminator_loss': disc_loss
    }).set_index('epoch')
    st.line_chart(df)


def plot_loss_curves(train_history, model_type, metric='loss'):
  if metric == 'fid':
    plot_fid(train_history)
  else:
    plot_combined_loss(train_history, model_type)