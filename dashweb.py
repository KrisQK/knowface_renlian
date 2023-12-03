import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_core_components as dcc

# 初始化 Dash 应用
app = dash.Dash(__name__)

# 第一页布局和内容
page_1_layout = html.Div([
    html.H1('Page 1'),
    dcc.Link('跳转到第二页', href='/page-2')
])

# 第二页布局和内容
page_2_layout = html.Div([
    html.H1('Page 2'),
    dcc.Link('返回第一页', href='/')
])

# 应用程序布局
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# 回调函数，根据 URL 路由加载不同的页面内容
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-2':
        return page_2_layout
    else:
        return page_1_layout

if __name__ == '__main__':
    app.run_server(debug=True)
