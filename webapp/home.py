# @Time : 2023/6/10 14:47 
# @Author : CaoXiang
# @Description:
import streamlit as st
import app_component as ac
from PIL import Image
st.set_page_config(layout="wide", page_title="Image Background Remover")
home_title = ":fire: 魔法衣橱"
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)
st.balloons()
st.markdown(":dog: **AI魔法衣橱通过上传一张衣服图像，实现AI自动生成模特上身效果！目前支持如下功能：**\n")
st.markdown(":star: **1. 局部换脸**")
st.text("根据一张模特图，更改模特面部，可以支持年龄、肤色、性别的定制化。")
st.markdown(":moon: **2. 生成模特**")
st.text("根据一张模特图，生成更多模特图，可以支持年龄、肤色、性别的定制化。")
st.markdown(":smile: **3. 更换背景**")
st.text("根据一张模特图，生成不同背景的图像")
st.markdown(":apple: **4. 款式裂变**")
st.text("根据一张模特图，自动裂变衣服的样式。")

with st.sidebar:
    ac.st_button(url="https://twitter.com/dclin", label="Let's connect", font_awesome_icon="fa-twitter")
    ac.st_button(url="https://www.buymeacoffee.com/gptlab", label="Buy me a coffee", font_awesome_icon="fa-coffee")
    ac.st_button(url="https://gptlab.beehiiv.com/subscribe", label="Subscribe to news and updates", font_awesome_icon="fa-newspaper-o")
    st.image(Image.open("/data/cx/ysp/aigc-smart-painter/ikun.jpg"), width=50)