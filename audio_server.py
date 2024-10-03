import argparse
import flask
import os
from flask import Flask, request, send_from_directory
from main import Runner


configs = {
    'batch_size':4096,  # 超参数batch大小
    'epoch':300,  # 一折的训练轮数
    'save_dir':"./audio_ResNet_V5_",  # 模型权重参数保存位置
    'lr':3e-5,
    'log_dir':'runs/audio_recog_V5'
}
"""
使用chrome进行测试时，如果服务端不是host在localhost上，会导致getUserMedia不可用（懒得搞https），所以要在信赖模式下启动chrome
"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --unsafely-treat-insecure-origin-as-secure="http://10.141.208.102" --user-data-dir="temp"
"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --unsafely-treat-insecure-origin-as-secure="http://10.141.208.102:22339" --user-data-dir="temp2"
"""


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    args = argparser.parse_args()

    app = flask.Flask(__name__, static_folder='interface')
    app.debug = True

    model = Runner(**configs)

    @app.route('/', methods=['POST', 'GET'])
    def home():
        return send_from_directory('interface', 'index.html')

    @app.route('/save-record', methods=['POST'])
    def save_record():
        file = flask.request.files['file']
        app.logger.debug(file.filename)
        os.makedirs("upload", exist_ok=True)
        save_to = "upload/{}".format(file.filename)
        file.save(save_to)
        result = model.test_single(path = save_to)
        return result

    @app.route('/js/<path:path>')
    def send_js(path):
        return send_from_directory('interface/js', path)


    app.run(host="0.0.0.0", port=22339)