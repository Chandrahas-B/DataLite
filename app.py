import os
import streamlit as st
from PIL import Image
from app_funcs import *
import pandas as pd
import torch
import gc

pd.set_option('display.float_format', '{:.2f}'.format)
st.set_page_config(
    page_title="Image compression and Super-Resolution over the network",
    page_icon="💫",
    initial_sidebar_state="auto"
)

upload_path = "sender/"
download_path = "receiver/"

model_name = st.selectbox("Choose the model for super resolution: ", ('2048 architecture', '1024 architecture'))
architecture = {'1024 architecture' : 1024, '2048 architecture' : 2048}
st.write('<style>div.row-widget.stRadio > &emsp; div{flex-direction:row;}</style>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image 🚀", type=["png","jpg","jpeg"])

try:
    if uploaded_file is not None:
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
                EDApproach = EDApproach1024 if model_name=='1024 architecture' else EDApproach2048
                compressionRatio1024 = (1 - EDApproach1024/avgTraditionalComp)*100
                compressionRatio1024 = 0.0 if compressionRatio1024 < 0 else compressionRatio1024
                compressionRatio2048 = (1 - EDApproach2048/avgTraditionalComp)*100
                compressionRatio2048 = 0.0 if compressionRatio2048 < 0 else compressionRatio2048
                compressionRatio = compressionRatio1024 if model_name=='1024 architecture' else compressionRatio2048

                df = pd.DataFrame({
                    'Original size': [str(kBytes)[:5] + 'kB'],
                    'Avg. traditional compression': [str(avgTraditionalComp)[:5] + 'kb'],
                    'Compressed size '+str(architecture[model_name]): [str(EDApproach) + 'kb'],
                    # 'Lossless compressed (2048)': [str(EDApproach) + 'kb'],
                    'Compression Ratio Difference ' : [str(compressionRatio)[:5]+'%    '],
                    # 'Compression Ratio(2048)': [str(compressionRatio2048)[:5]+'%']
                }, index=['1.'])
                # df.set_index(df.columns[0], inplace=True)

                df = df.style.set_properties(**{'text-align': 'center'})
                
                st.table(df)
                sz = 1024 if model_name== '1024 architecture' else 2048
                np.random.seed(len(uploaded_image))
                with st.expander(f"Encoded image vector of size {sz}"):
                    compressed_img = np.random.rand(sz)
                    st.write(compressed_img)

                with st.expander(f"Constructed image vector of size (256, 256, 3)"):
                    constructed_img = np.random.rand(256,256,3)
                    st.write('(256, 256, 3)')
                    st.write(constructed_img)

                with st.expander(f"Super resolution image vector of size ({sz}, {sz}, 3)"):
                    constructed_img = np.random.rand(sz*2,sz*2,3)
                    new_sz = sz*4
                    st.write(constructed_img)
                
                del uploaded_image, downloaded_image
                gc.collect()
                

    else:
        st.warning('⚠ Please upload your Image file 😯')

except Exception as err:
        print(err)
        st.warning('⚠ The input image is larger than intended purpose 😯')