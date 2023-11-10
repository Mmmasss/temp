var app = new Vue({
    el: '#app',
    data: {
        map: null, //地图对象
        ruler: null, //测距工具
        border: null, //地图边界线
        query_trajectory: null, // 查询轨迹
        query_trajectory_info: {
            start_coord: null,
            end_coord: null,
            length: 0,
            distance: 0,
        }, // 当前Query轨迹信息栏
        ruler_markers: [],
        query_form: {
            query_type: 'efficient_bf', // 默认选择高效-暴力
            time_range_on: false, // 默认不选择时间范围
            time_range: [new Date(2016, 11, 1, 6, 0),
                new Date(2016, 11, 1, 23, 45)],
            query_k: 10, // 默认选择k=10
        }, //查询表单
        response_info: {
            compute_time: 0,
            query_time: 0,
            compute_count: 0,
            query_type: '',
        }, //查询结果信息栏
        result_trajectories: [], // 查询结果:放的是createTrajectory创建的对象的列表
        result_trajectories_table: [], //查询结果表格:放的是一堆json格式数据
        query_trajectory_opts: {
            strokeColor: "red", //线颜色
            strokeOpacity: 0.8,
            strokeWeight: 10,
            zIndex: 10,
        }, //查询轨迹样式
        result_trajectory_opts: {
            strokeColor: "#39c5bb", //线颜色
            strokeOpacity: 0.5,
            strokeWeight: 8,
            zIndex: 9,
        }, //查询结果轨迹样式
        select_options: [{
            value: 'efficient_bf',
            label: '高效-暴力'
        }, {
            value: 'efficient_faiss',
            label: '高效-Faiss'
        }, {
            value: 'discret_frechet',
            label: 'Frechet距离'
        }, {
            value: 'hausdorff',
            label: 'Hausdorff距离'
        }, {
            value: 'lcss',
            label: 'LCSS'
        }, {
            value: 'sspd',
            label: 'SSPD'
        }],
        loading: false, //控制表格加载状态
        activeNames: ['1'] //控制展开哪个tab
    },
    mounted() {
        this.map = new AMap.Map("container", {
            center: [104.08275, 30.67225],
            zoom: 15
        });
        var startMarkerOptions = {
            icon: new AMap.Icon({
                size: new AMap.Size(19, 31),//图标大小
                imageSize: new AMap.Size(19, 31),
                image: "https://webapi.amap.com/theme/v1.3/markers/b/start.png"
            })
        };
        var endMarkerOptions = {
            icon: new AMap.Icon({
                size: new AMap.Size(19, 31),//图标大小
                imageSize: new AMap.Size(19, 31),
                image: "https://webapi.amap.com/theme/v1.3/markers/b/end.png"
            }),
            offset: new AMap.Pixel(-9, -31)
        };
        var midMarkerOptions = {
            icon: new AMap.Icon({
                size: new AMap.Size(19, 31),//图标大小
                imageSize: new AMap.Size(19, 31),
                image: "https://webapi.amap.com/theme/v1.3/markers/b/mid.png"
            }),
            offset: new AMap.Pixel(-9, -31)
        };
        var rulerOptions = {
            startMarkerOptions: startMarkerOptions,
            midMarkerOptions: midMarkerOptions,
            endMarkerOptions: endMarkerOptions,
            lineOptions: this.query_trajectory_opts
        };
        this.ruler = new AMap.RangingTool(this.map, rulerOptions);
        this.ruler.on("end", function (event) {
            console.log(event);
            app.query_trajectory = event.polyline
            app.query_trajectory_info.start_coord = "(" + event.points[0].lng.toFixed(3).toString() +
                ", " + event.points[0].lat.toFixed(3).toString() + ")"
            app.query_trajectory_info.end_coord = "(" + event.points[event.points.length - 1].lng.toFixed(3).toString()
                + ", " + event.points[event.points.length - 1].lat.toFixed(3).toString() + ")"
            app.query_trajectory_info.distance = event.distance
            app.query_trajectory_info.length = app.ruler_markers.length
            app.ruler.turnOff()
        })
        this.ruler.on("addnode", function (event) {
            console.log(event);
            app.ruler_markers.push(event.marker)
        })
        this.border = this.createBorder();
    },
    methods: {
        startDrawTraj() {
            if (this.query_trajectory != null) {
                this.clearDrawTraj()
            }
            this.ruler.turnOn()
        },
        clearDrawTraj() {
            this.map.remove(this.query_trajectory)
            this.map.remove(this.ruler_markers)
            $(".amap-ranging-label").remove()
            this.query_trajectory = null;
            this.ruler_markers = [];
            this.ruler.turnOff()
        },
        createBorder() {
            var pad = 0.002
            var max_lon = 104.12958 - pad
            var max_lat = 30.72775 - pad
            var min_lon = 104.04214 + pad
            var min_lat = 30.65294 + pad
            return new AMap.Polyline({
                map: this.map,
                path: [
                    [min_lon, min_lat],
                    [min_lon, max_lat],
                    [max_lon, max_lat],
                    [max_lon, min_lat],
                    [min_lon, min_lat]
                ],
                strokeColor: "#000000",
                strokeOpacity: 0.8,
                strokeWeight: 4,
                strokeStyle: "dashed",
                strokeDasharray: [10, 5],
            })
        },
        createTrajectory(tid, points, polyline_opts) {
            polyline_opts['path'] = points
            var polyline = new AMap.Polyline(polyline_opts);
            var start_marker = new AMap.Marker({
                position: new AMap.LngLat(points[0][0], points[0][1]),
                title: tid + '起点',
            });
            var end_marker = new AMap.Marker({
                position: new AMap.LngLat(points[points.length - 1][0], points[points.length - 1][1]),
                title: tid + '终点',
            });
            return {
                "polyline": polyline,
                "start_marker": start_marker,
                "end_marker": end_marker,
            }
        },
        query() {
            if (this.query_trajectory === null) {
                app.$message({type: 'error', message: '要查询的轨迹为空'});
            } else {
                this.activeNames = ["3"]
                app.loading = true;
                const type = this.query_form.query_type
                let time_range = null
                if (this.query_form.time_range_on) {
                    time_range = [this.query_form.time_range[0].getTime() / 1000, this.query_form.time_range[1].getTime() / 1000]
                }
                const st = Date.now()
                $.ajax({
                    type: "POST",
                    url: "/query",
                    data: {
                        query_traj: JSON.stringify(this.query_trajectory.$x[0]),
                        query_type: type,
                        time_range: JSON.stringify(time_range),
                        k: this.query_form.query_k,
                    },
                    success: function (data) {
                        console.log(data)
                        app.loading = false;
                        if (data.success) {
                            app.$message({type: 'success', message: '查询成功!'});
                            app.response_info.query_type = type
                            app.response_info.compute_time = data.result.compute_time.toFixed(2)
                            app.response_info.query_time = (Date.now() - st) / 1000
                            app.response_info.compute_count = data.result.compute_count
                            traj_list = data.result.traj_list
                            for (traj of traj_list) {
                                traj.start_time = app.getLocalTime(traj.start_time)
                                traj.end_time = app.getLocalTime(traj.end_time)
                                traj.sim = traj.sim.toFixed(5)
                            }
                            app.result_trajectories_table = data.result.traj_list
                            app.drawResult();
                        } else {
                            app.$message({type: 'error', message: '查询失败'});
                        }
                    }

                })
            }
        },
        drawResult() {
            var new_result_trajectories = []
            for (traj of this.result_trajectories_table) {
                var opts = JSON.parse(JSON.stringify(this.result_trajectory_opts))
                var trajectory = this.createTrajectory(traj.tid, traj.points, opts)
                new_result_trajectories.push(trajectory)
            }
            this.result_trajectories = new_result_trajectories
        },
        handleHighlight(index, row) {
            console.log(index)
            console.log(row)
        },
        toStepTwo() {
            if (this.query_trajectory === null) {
                app.$message({type: 'error', message: '请先选择轨迹'});
            } else {
                this.activeNames = ["2"]
                $(".amap-ranging-label").remove()
            }
        },
        tableRowClassName({row, rowIndex}) {
            row.row_index = rowIndex;
        },
        cellClick(row, column, cell, event) {
            var polyline = this.result_trajectories[row.row_index].polyline
            this.changeHighlightStatus(polyline)
        },
        getLocalTime(timestamp) {
            var date = new Date(timestamp * 1000);
            var YY = date.getFullYear() + '-';
            var MM = (date.getMonth() + 1 < 10 ? '0' + (date.getMonth() + 1) : date.getMonth() + 1) + '-';
            var DD = (date.getDate() < 10 ? '0' + (date.getDate()) : date.getDate());
            var hh = (date.getHours() < 10 ? '0' + date.getHours() : date.getHours()) + ':';
            var mm = (date.getMinutes() < 10 ? '0' + date.getMinutes() : date.getMinutes()) + ':';
            var ss = (date.getSeconds() < 10 ? '0' + date.getSeconds() : date.getSeconds());
            return hh + mm + ss;
        },
        changeHighlightStatus(polyline) {
            if (polyline.getOptions().zIndex !== 99) {
                polyline.setOptions({
                    strokeOpacity: 1,
                    strokeWeight: 15,
                    zIndex: 99,
                },)
                console.log("highlight")
            } else {
                polyline.setOptions(this.result_trajectory_opts)
                console.log("set back to original")
            }
        },
    }
})
app.$watch('result_trajectories', function (newVal, oldVal) {
    for (traj of oldVal) {
        this.map.remove(traj.polyline)
        this.map.remove(traj.start_marker)
        this.map.remove(traj.end_marker)
    }
    for (traj of newVal) {
        this.map.add(traj.polyline)
        this.map.add(traj.start_marker)
        this.map.add(traj.end_marker)
    }
})