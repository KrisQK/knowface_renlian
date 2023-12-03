import streamlit as st

# 添加标题
st.title('我的第一个 Streamlit 应用')

# 添加文本
st.write('这是一个文本部分')

# 创建一个交互式小部件
user_input = st.text_input('请输入您的名字', '在此输入您的名字...')
st.write('您输入的名字是:', user_input)
