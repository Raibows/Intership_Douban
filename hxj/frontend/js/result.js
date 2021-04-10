// 获取地址栏参数
function GetQueryString(value) {
    var url = decodeURI(location.search);
    var object = {};
    if (url.indexOf("?") != -1) {
        var str = url.substr(1);
        var strs = str.split("&");
        for (var i = 0; i < strs.length; i++) {
            object[strs[i].split("=")[0]] = strs[i].split("=")[1];
        }
    }
    return object[value];
}

// 获取类别并进行存储
var categories = new Array();
var colorCate = new Map();
// 随机生成颜色
function getColor() {
    var colorElements = "0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f";
    var colorArray = colorElements.split(",");
    var color = "#";
    for (var i = 0; i < 6; i++) {
        color += colorArray[Math.floor(Math.random() * 16)];
    }
    return color;
}


$(function() {
    getCategories(); // 获取类别并设置颜色
    getData(); // 获取数据渲染
})

function getCategories() {
    $.ajax({
        type: "get",
        url: "http://127.0.0.1:8000/type",
        dataType: "json",
        success: function(response) {
            if (response.code != 200) {
                alert("请求失败！请检查");
                return;
            }
            data = response.data;
            for (var t in data) {
                colorCate[t] = getColor()
                categories.push({
                    "name": t,
                    "index": data[t]
                })
            }
        },
    });
}

function getData() {
    var movieName = GetQueryString("name");
    console.log(movieName);
    // 赋予默认的name
    if (movieName == undefined) {
        movieName = "人潮汹涌";
    }
    // 修改title和名字
    $("title").text(movieName);
    $("#name").text(movieName);
    $("#title").text(movieName);

    // 获取数据显并进行显示
    $.ajax({
        type: "get",
        url: "http://127.0.0.1:8000/movie/" + movieName,
        data: {
            name: movieName,
        },
        dataType: "json",

        success: function(response) {
            if (response.code != 200) {
                alert("请求失败！请检查");
                return;
            }
            data = response.data;
            $("#img_url").attr("src", data["pic_url"]);
            $("#director").text(data["director"]);
            $("#rate").text(data["rate"] + "⭐");
            $("#genres").text(data["genres"]);
            $("#imdb").text(data["imdb"]);
            $("#imdb").attr("href", "https://www.imdb.com/title/" + data["imdb"]);
            $("#page_url").text(data["page_url"]);
            $("#region").text(data["region"]);
            $("#brief").text(data["brief"]);

            // 遍历显示相似影片
            // var targetsContainer = $("#targetsContainer");
            var targetsContainer = $("#targetsContainer");
            targetsContainer.html("");
            // var targetData = ;
            for (let index = 0; index < data["targets_info"].length; index++) {
                const targetData = data["targets_info"][index]
                targetDataGenres = targetData["genres"].split("/")
                const element = `<div class="feature">
                                <div class="feature-inner targets">
                                    <div class="feature-icon">
                                        <img src="${targetData["pic_url"]}" alt="${targetData["name"]}">
                                    </div>
                                    <h4 class="feature-title h3-mobile">${targetData["name"]}</h4>
                                    <p class="text-sm">
                                        ${genresCreate(targetDataGenres)}
                                    </p>
                                </div>
                            </div>`;
                targetsContainer.append(element);
            }

            // 创建分类概率图节点
            var data = nodeCreate(data);
            graphCreate(data);
        },
    });
}

// 为类别创建不同颜色标签
function genresCreate(targetDataGenres) {
    var content = '';
    for (let i = 0; i < targetDataGenres.length; i++) {
        const element = targetDataGenres[i];
        content += `<span class="badge" style="background:${colorCate[element]}">${element}</span>`;
    }
    return content;
}


function nodeCreate(data) {
    var nodeList = []
    var linkList = []

    var name = data["name"]
    nodeList.push({ name: name, symbolSize: 70, category: 000 });
    // 生成类别节点和链接
    for (let i = 0; i < categories.length; i++) {
        const cate_g = categories[i];
        nodeList.push({
            name: cate_g["name"],
            symbolSize: 50,
            category: cate_g["index"],
        });
        // 检查当前类别是否是该电影所属类别，确定概率的值
        if ($.inArray(cate_g["index"], data["genres_index"]) != -1) {
            linkList.push({
                source: name,
                target: cate_g["name"],
                name: Math.random(0.5, 1),
            });
        } else {
            linkList.push({
                source: name,
                target: cate_g["name"],
                name: Math.random(0, 0.3),
            });
        }

    }
    return { "nodeList": nodeList, "linkList": linkList };
}


// 关系图绘制
function graphCreate(data) {
    var myChart = echarts.init(document.getElementById("graph"));
    console.log(myChart)
        // 从接口加载数据，加载类别

    option = {
        // 图的标题
        title: {
            text: "分类概率图",
        },
        // 提示框的配置
        tooltip: {
            formatter: function(x) {
                return x.data.des;
            },
        },
        // 工具箱
        toolbox: {
            // 显示工具箱
            show: true,
            feature: {
                mark: {
                    show: true,
                },
                // 还原
                restore: {
                    show: true,
                },
                // 保存为图片
                saveAsImage: {
                    show: true,
                },
            },
        },
        //图例配置
        legend: [{
            // selectedMode: 'single',
            data: categories.map(function(a) {
                return a.name;
            }),
        }, ],
        series: [{
            type: "graph", // 类型:关系图
            layout: "force", //图的布局，类型为力导图
            symbolSize: 40, // 调整节点的大小
            roam: true, // 是否开启鼠标缩放和平移漫游。默认不开启。如果只想要开启缩放或者平移,可以设置成 'scale' 或者 'move'。设置成 true 为都开启
            focusNodeAdjacency: true,
            edgeSymbol: ["circle", "none"], //rrow/none
            edgeSymbolSize: [2, 10],
            edgeLabel: {
                normal: {
                    textStyle: {
                        fontSize: 20,
                    },
                },
            },
            force: {
                repulsion: 2500,
                edgeLength: [10, 50],
                gravity: 0.2,
                layoutAnimation: true, //是否关闭加载时的动画，如果关闭可能会在刷新节点布局后缩放造成错位
                friction: 1, //减缓节点的移动速度
            },
            draggable: true,
            lineStyle: {
                normal: {
                    width: 1,
                    color: "#4b565b",
                },
            },
            edgeLabel: {
                normal: {
                    show: true,
                    formatter: function(x) {
                        return x.data.name;
                    },
                },
            },
            label: {
                normal: {
                    show: true,
                    textStyle: {},
                },
            },

            // 数据
            data: data.nodeList,
            links: data.linkList,
            categories: categories,
        }, ],
    };

    myChart.setOption(option);
    //点击事件
    // myChart.on("click", function(params) {
    //     alert(params.name);
    // });
}