function preload() {
    q_viewer.load.spritesheet('poker', 'static/i/poker.png', 90, 120);
}

function create() {
    q_viewer.stage.setBackgroundColor('#71c5cf');
    q_viewer.camera.focusOnXY(0, 40);
    // console.log('on create');
    // var p = new PG.Poker(q_viewer, 0, 0);
    // q_viewer.world.add(p);
    // q_viewer.add.sprite(q_viewer.world.width / 2, q_viewer.world.height * 0.4, 'poker', 0);
    // q_viewer.add.tween(p).to({
    //     x: q_viewer.world.width / 2,
    //     y: q_viewer.world.height - 100
    // }, 500, Phaser.Easing.Default, true, 50);
}
// var h = window.innerHeight * 960/window.innerWidth;
// if (h > 540) h = 540;
var q_viewer = new Phaser.Game(630, 500, Phaser.AUTO, 'vis_cards', {preload: preload, create: create});

var width = null;
var myChart = null;

function init_charts() {
    width = document.getElementById('q_values').offsetWidth;
    document.getElementById('q_values').style.width = width + 'px';
    myChart = echarts.init(document.getElementById('q_values'));
}

function draw_comb(combinations) {
    return;
    if (myChart == null) {
        init_charts();
    }
    q_viewer.world.removeAll();
    var q_values = [];
    for (var i = 0; i < combinations.length; i++) {
        q_values.push(combinations[i][1]);
        var y_offset = 60 + i * (q_viewer.world.height - 120) / (combinations.length - 1);
        if (combinations.length == 1) {
            y_offset = 60;
        }
        var x_offset = 45;
        var cnts = 0;
        for (var j = 0; j < combinations[i][0].length; j++) {
            cnts += combinations[i][0][j].length + 1;
        }
        // cnts -= 1;
        var x_spacing = (q_viewer.world.width - 90) / (cnts - 1);
        cnts = 0;
        for (j = 0; j < combinations[i][0].length; j++) {
            if (combinations[i][0][j].length == 0) {
                var p = new Phaser.Sprite(q_viewer, x_offset + cnts * x_spacing, y_offset, 'poker', 64);
                p.anchor.set(0.5);
                q_viewer.world.add(p);
                cnts += 1;
            }
            for (var k = 0; k < combinations[i][0][j].length; k++) {
                var p = new Phaser.Sprite(q_viewer, x_offset + cnts * x_spacing, y_offset, 'poker', combinations[i][0][j][k]);
                p.anchor.set(0.5);
                q_viewer.world.add(p);
                cnts += 1;
            }
            cnts += 1;
        }
    }

    var option = {
        title: {
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        legend: {
            data: ['Q value']
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            boundaryGap: [0, 0.01]
        },
        yAxis: {
            type: 'category',
            data: ['top1', 'top2', 'top3', 'top4', 'top5'].slice(0, q_values.length).reverse()
        },
        series: [
            {
                name: 'Q value',
                type: 'bar',
                data: q_values.reverse()
            }
        ]
    };

    myChart.setOption(option);
}

function draw_fine(groups) {
    return;
    if (myChart == null) {
        init_charts();
    }
    q_viewer.world.removeAll();
    var q_values = [];
    for (var i = 0; i < groups.length; i++) {
        q_values.push(groups[i][1]);
        var y_offset = 60 + i * (q_viewer.world.height - 120) / (groups.length - 1);
        if (groups.length == 1) {
            y_offset = 60;
        }
        if (groups[i][0].length == 0) {
            var x_offset = 45;
            var p = new Phaser.Sprite(q_viewer, x_offset, y_offset, 'poker', 64);
            p.anchor.set(0.5);
            q_viewer.world.add(p);
            continue;
        }
        for (var j = 0; j < groups[i][0].length; j++) {
            var x_offset = 45 + 33 * j;
            var p = new Phaser.Sprite(q_viewer, x_offset, y_offset, 'poker', groups[i][0][j]);
            p.anchor.set(0.5);
            q_viewer.world.add(p);
        }
    }

    var option = {
        title: {
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        legend: {
            data: ['Q value']
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            boundaryGap: [0, 0.01]
        },
        yAxis: {
            type: 'category',
            data: ['top1', 'top2', 'top3', 'top4', 'top5'].slice(0, q_values.length).reverse()
        },
        series: [
            {
                name: 'Q value',
                type: 'bar',
                data: q_values.reverse()
            }
        ]
    };

    myChart.setOption(option);
}
