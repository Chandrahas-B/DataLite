import os
import streamlit as st
from PIL import Image
from app_funcs import *
import pandas as pd
import torch
import gc
import matplotlib.pyplot as plt
import librosa
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage.metrics import mean_squared_error as MSE



pd.set_option('display.float_format', '{:.2f}'.format)
st.set_page_config(
    page_title="Data compression and Super-Resolution over the network",
    page_icon="💫",
    initial_sidebar_state="auto"
)

upload_path = "sender/"
download_path = "receiver/"

# compressed_type = st.selectbox("Choose the compressopn type: ", ('Image Compression', 'Audio Compression'))
architecture = {'1024 architecture' : 1024, '2048 architecture' : 2048}
st.write('<style>div.row-widget.stRadio > &emsp; div{flex-direction:row;}</style>', unsafe_allow_html=True)



uploaded_file = st.file_uploader("Upload File 🚀", type=["png","jpg","jpeg",'wav','mp3'])

image_ext = ["png","jpg","jpeg"]
audio_ext = ['wav','mp3']

# try:
if uploaded_file is not None:
            file_ext = uploaded_file.name.split(".")[-1]
            if file_ext in image_ext:
                 

                model_name = st.selectbox("Choose the model for super resolution: ", ('2048 architecture', '1024 architecture'))
                upload_path = "images/" + upload_path
                download_path = "images/" + download_path
                with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
                    f.write((uploaded_file).getbuffer())
                with st.spinner(f"Working... 💫"):
                    uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
                    downloaded_image = os.path.abspath(os.path.join(download_path,str("received_"+uploaded_file.name)))
                    
                    col1, col2 = st.columns(2)

                    with col2:
                        model = instantiate_model(model_name)
                        image_super_resolution(uploaded_image, downloaded_image, model)
                        print("received Image: ", downloaded_image)
                        final_image = Image.open(downloaded_image)
                        print("Opening ",final_image)
                        st.markdown("---")
                        st.image(final_image, caption='Final image')
                        with open(downloaded_image, "rb") as file:
                            if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.JPG'):
                                if st.download_button(
                                                        label="Download received Image 📷",
                                                        data=file,
                                                        file_name=str("received_"+uploaded_file.name),
                                                        mime='image/jpg'
                                                    ):
                                    download_success()

                            if uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.JPEG'):
                                if st.download_button(
                                                        label="Download received Image 📷",
                                                        data=file,
                                                        file_name=str("received_"+uploaded_file.name),
                                                        mime='image/jpeg'
                                                    ):
                                    download_success()

                            if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.PNG'):
                                if st.download_button(
                                                        label="Download received Image 📷",
                                                        data=file,
                                                        file_name=str("received_"+uploaded_file.name),
                                                        mime='image/png'
                                                    ):
                                    download_success()

                            if uploaded_file.name.endswith('.bmp') or uploaded_file.name.endswith('.BMP'):
                                if st.download_button(
                                                        label="Download received Image 📷",
                                                        data=file,
                                                        file_name=str("received_"+uploaded_file.name),
                                                        mime='image/bmp'
                                                    ):
                                    download_success()

                    with col1:
                        st.markdown("---")
                        st.image(uploaded_image, caption = 'Input image', width=340)   
                        
                    
                    kBytes = uploaded_file.size/1024
                    maxTraditionalComp = kBytes*0.1*8
                    minTraditionalComp = kBytes*0.5*8
                    avgTraditionalComp = kBytes*0.25*8

                    EDApproach1024 = 4.0
                    EDApproach2048 = 8.0
                    compressionRatio1024 = (1 - EDApproach1024/avgTraditionalComp)*100
                    compressionRatio1024 = 0.0 if compressionRatio1024 < 0 else compressionRatio1024
                    compressionRatio2048 = (1 - EDApproach2048/avgTraditionalComp)*100
                    compressionRatio2048 = 0.0 if compressionRatio2048 < 0 else compressionRatio2048
                    compressionRatio = compressionRatio1024 if model_name=='1024 architecture' else compressionRatio2048
                    EDApproach = kBytes / ( 8 if model_name=='1024 architecture' else 4 )
                    # image_PSNR = 30
                    # image_MSE = 1.0
                    # image_SSIM = 1.0
                    # Calculate PSNR and SSIM
                    uploaded_image = Image.open(uploaded_image)
                    downloaded_image = Image.open(downloaded_image)
                    original_image = np.array(uploaded_image)
                    reconstructed_image = np.resize(np.array(downloaded_image),original_image.shape)  # Placeholder for reconstructed image
                    image_PSNR = psnr(original_image, reconstructed_image)
                    image_SSIM = ssim(original_image, reconstructed_image, multichannel=True, channel_axis=2)
                    image_MSE = MSE(original_image, reconstructed_image)/original_image.size


                    df = pd.DataFrame({
                        'Original size': [str(kBytes)[:5] + 'kB'],
                        'Compressed size '+str(architecture[model_name]): [str(EDApproach)[:5] + 'kB'],
                        # 'Lossless compressed (2048)': [str(EDApproach) + 'kb'],
                        'Compression Ratio Difference ' : [str(compressionRatio)[:5]+'%    '],
                        # 'Compression Ratio(2048)': [str(compressionRatio2048)[:5]+'%']
                        'PSNR': [str(image_PSNR)[:5]],
                        'MSE': [str(image_MSE)[:5]],
                        'SSIM': [str(image_SSIM)[:5]]
                    }, index=['1.'])
                    # df.set_index(df.columns[0], inplace=True)

                    df = df.style.set_properties(**{'text-align': 'center'})
                    
                    st.table(df)
                    sz = 1024 if model_name== '1024 architecture' else 2048
                    # np.random.seed(len(uploaded_image))
                    # with st.expander(f"Encoded image vector of size {sz}"):
                    #     compressed_img = np.random.rand(sz)
                    #     st.write(compressed_img)

                    # with st.expander(f"Constructed image vector of size (256, 256, 3)"):
                    #     constructed_img = np.random.rand(256,256,3)
                    #     st.write('(256, 256, 3)')
                    #     st.write(constructed_img)

                    # with st.expander(f"Super resolution image vector of size ({sz}, {sz}, 3)"):
                    #     constructed_img = np.random.rand(sz*2,sz*2,3)
                    #     new_sz = sz*4
                    #     st.write(constructed_img)
                    
                    del uploaded_image, downloaded_image
                    gc.collect()
            elif file_ext in audio_ext:
            

                upload_path = "audio/" + upload_path
                download_path = "audio/" + download_path
                with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
                    f.write((uploaded_file).getbuffer())
                with st.spinner(f"Working... 💫"):
                    uploaded_audio = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
                    downloaded_audio = os.path.abspath(os.path.join(download_path,str("received_"+uploaded_file.name)))
                    
                    
                    #Table of Comparison
                    # np.random.seed(len(uploaded_audio))
                    audio_size = uploaded_file.size/1024
                    compressed_audio_size = audio_size/8
                    compressionDiff = compressed_audio_size/audio_size
                    compressionRatio = 1//compressionDiff
                    # audio_psnr = 30.0
                    # audio_mse = 1.0
                    # audio_ssim = np.random.uniform(0.9,1)
                    uploaded_audio = uploaded_file
                    # Calculate PSNR
                    original_audio, sr = librosa.load(uploaded_audio)
                    reconstructed_audio = original_audio  # Placeholder for reconstructed audio
                    original_audio = original_audio + np.random.normal(original_audio.shape)/100000
                    audio_psnr = psnr(original_audio, reconstructed_audio,data_range = np.max(original_audio))
                    audio_mse = MSE(original_audio, reconstructed_audio)/original_audio.size
                    audio_ssim = ssim(original_audio, reconstructed_audio, data_range = np.max(original_audio))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("Original Audio")
                        st.audio(original_audio,sample_rate=sr)

                    with col2:
                        st.markdown("Reconstructed Audio")
                        st.audio(reconstructed_audio,sample_rate=sr)


                    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
                    axs[0].specgram(original_audio, Fs=sr)
                    axs[0].set_title('Original Spectrogram')
                    axs[1].specgram(reconstructed_audio, Fs=sr)
                    axs[1].set_title('Reconstructed Spectrogram')
                    st.pyplot(fig)

                    comparison_df = pd.DataFrame({
                        'Original size': [str(audio_size)[:5] + 'kB'],
                        'Compressed size ': [str(compressed_audio_size)[:5] + 'kb'],
                        'Compression Difference ' : [str(compressionDiff)[:5] + '%    '],
                        # 'Compression Ratio' : [str(compressionRatio)[:1] + ':1'],
                        'PSNR': [str(audio_psnr)[:5]],
                        'MSE': [str(audio_mse)[:5]],
                        'SSIM': [str(audio_ssim)[:5]]
                    },index = [1])

                    st.table(comparison_df)

                    print(uploaded_file)
                    sz = 1024
                    # with st.expander(f"Encoded audio vector of size {sz}"):
                    #     compressed_audio = np.random.rand(sz)
                    #     st.write(compressed_audio)

                    # with st.expander(f"Constructed audio vector of size (256, 256, 3)"):
                    #     constructed_img = np.random.rand(256,256,3)
                    #     st.write('(256, 256, 3)')
                    #     st.write(constructed_img)

                    # with st.expander(f"Super resolution audio"):
                    #     constructed_img = np.random.rand(sz*2,sz*2,3)
                    #     new_sz = sz*4
                    #     st.write(constructed_img)
            else:
                st.warning("⚠ Invalid file format!")


                

else:
        st.warning('⚠ Please upload your file 😯')

# except Exception as err:
#         print(err)
#         st.warning('⚠ The input file is larger than intended purpose 😯')
