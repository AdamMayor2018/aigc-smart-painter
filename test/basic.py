# @Time : 2023/6/11 16:22 
# @Author : CaoXiang
# @Description:
import streamlit as st
from PIL import Image
# text/Title
st.set_page_config(layout="wide", page_title="Image Background Remover")
st.title("AI 自动换装效果展示")
st.balloons()
st.sidebar.header("功能导航栏")
st.sidebar.write("## Upload and download :gear:")
my_upload1 = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="1")
st.sidebar.write("## Upload and download :gear:")
my_upload2 = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="2")
# Header/Subheader
#st.text("上传一张图像，实现AI自动生成模特上身效果！")
st.markdown(":dog: **上传一张衣服图像，实现AI自动生成模特上身效果！**")
# st.info/success/st.warning/st.error/st.exception 分别对应不同颜色
st.info("上传的图像需要是假人或者真人模特的真实拍摄照片!如下所示：")
img = Image.open("/data/cx/datasets/clothes/女装1.jpg")
st.image(img, caption="举例图像", width=300)

#video/audio
# vid_file = open("example.mp4", "rb").read()
# st.video(vid_file)
# audio_file = open("example.mp3", "rb").read()
# st.audio(audio_file, format="audio/mp3")

#checkbox
# if st.checkbox("show/hide"):
#     st.text("showing or hiding wigets.")

#selectbox
#occupation = st.selectbox("Your Occupation", ["Programmer", "DataScientist", "Doctor", "Businessman"])
#st.write("You selected this option", occupation)


#radio
# status_task = st.radio("1.选择要做的任务：", ("模特换脸", "服装裂变", "更换背景"))
# if status_task == "模特换脸":
#     st.text("随机更换模特的脸部，保持姿态不变。")
# elif status_task == "服装裂变":
#     st.text("随机更换模特的服装，保持姿态不变。")
# else:
#     st.text("随机更换模特的背景，保持姿态不变。")

#multiselect
status_task = st.multiselect("1.选择要做的任务：", ("模特换脸", "服装裂变", "更换背景"), default="模特换脸")
st.text(f"已选定的任务：{status_task}")

status_num = st.radio("2.选择要做的图像数量：", (1, 2, 4, 8))
st.text("选择要做的图像数量，最多支持8张图像同时处理。")

#slider
age = st.slider("3.选择模特的年龄：", 10, 100)

#select_box
continets = st.selectbox("4.选择模特的类型：", ["欧美", "中国", "日韩", "非洲"])

addition_prompt = st.text_input("5.输入额外的提示语, 用逗号分隔", "请在此输入")
#addition_prompt = st.text_area("5.输入额外的提示语, 用逗号分隔", "请在此输入")
# if addition_prompt:
#     st.text(f"获取到额外的提示语：{addition_prompt}")
#button
if st.button("submit"):
    st.success(f"获取到额外的提示语：{addition_prompt}")

# Date Input
# import datetime
# today = st.date_input("6.选择日期", datetime.datetime.now())

# Time Input
# the_time = st.time_input("7.选择时间", datetime.time())


#Display raw code
# st.code("import numpy as np")
#
# #Display raw code
# with st.echo():
#     # This will also be shown
#     import pandas as pd
#     df = pd.DataFrame()

#Display progress bar
import time
my_bar = st.progress(0)
for p in range(100):
    my_bar.progress(p + 1)
    time.sleep(0.1)

#Dataframe
# st.dataframe(df)
#
# #Table
# st.table(df)