# Model for inferencing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.models import load_model
import tensorflow as tf

# Generator 
def build_generator():
    model = Sequential()
    
    model.add(Dense(7*7*128, input_dim= 128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding= 'same'))
    model.add(LeakyReLU(0.2))
    
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding= 'same'))
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2D(128, 4, padding= 'same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding= 'same'))
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2D(1, 4, padding= 'same', activation= 'sigmoid'))
    
    return model

model = build_generator()
model.compile()
model.load_weights('generator_weights.h5')

# Streamlit App
import streamlit as st
import matplotlib.pyplot as plt

# to use the whole width of the page
st.set_page_config(layout='wide')

# to hide the menu button
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.write('''
# Image Generation GAN

### This App Generates Random Fashion Images using generative adversarial Neural Networks.
The model is trained for 400 epochs which took over 9 hours to train with GPU.  
**Dataset:** [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)  
**Code:** [Image Generation using GAN](https://www.kaggle.com/code/apurvayadav29/image-generation-using-gan/notebook)  
  
***
''')

st.write('Press the below button to Generate 16 Random Fashion Images')
if st.button('Generate Images'):
    model_img = model.predict(tf.random.normal((16, 128, 1)))
    fig, ax = plt.subplots(ncols= 4, nrows=4, figsize=(10,10))
    
    for r in range(4):
        for c in range(4):
            ax[r][c].imshow(model_img[(r+1)*(c+1)-1]).axes.axis('off')
            
    
    
    plt.show()
    st.pyplot(fig, clear_figure= True)


st.write("***")
expand_bar = st.expander('About')
expand_bar.markdown('''
GitHub: [Image-Generation-GAN](https://github.com/apurvayadav/Image-Generation-GAN)  
Reference: [Build a Generative Adversarial Neural Network with Tensorflow and Python | Deep Learning Projects](https://youtu.be/AALBGpLbj6Q)
''')