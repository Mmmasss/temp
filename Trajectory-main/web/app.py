from flask import Flask, jsonify, render_template, request
import json
from datetime import timedelta
from service import Service
import model

app = Flask(__name__)
app.jinja_env.variable_start_string = '[['  # 解决jinja2和vue的分隔符{{}}冲突
app.jinja_env.variable_end_string = ']]'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)  # 浏览器不缓存实时更新静态文件
service = Service()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query', methods=['GET', 'POST'])
def query():
    query_traj = json.loads(request.form.get("query_traj"))
    query_type = request.form.get("query_type")
    time_range = json.loads(request.form.get("time_range"))
    k = int(request.form.get("k"))
    # time_range = [1478063519, 1478064044]
    # time_range = None
    # k = 3
    traj_list, sim_list, compute_time, compute_count = service.knn_query(query_traj, query_type, k, time_range)
    for i in range(len(traj_list)):
        traj_list[i] = traj_list[i].to_json()
        traj_list[i]['sim'] = sim_list[i]
        traj_list[i].pop('embedding')
    result = {"traj_list": traj_list, "compute_time": compute_time, "compute_count": compute_count}
    return jsonify({"code": 200, "success": True, "result": result, "msg": "查询成功"})


if __name__ == '__main__':
    app.run(debug=True)
