var c = document.getElementById("canvas");
var ctx = c.getContext("2d");
var image = new Image();
var dimension = 256;
console.log(`Label changed ${changed}`)

var drawBoxes = function(id, xMin, xMax, yMin, yMax) {
    ctx.strokeStyle = "red";
    ctx.fillStyle = "red";
    ctx.rect(xMin, yMin, xMax - xMin, yMax - yMin);
    ctx.lineWidth="2";
    ctx.stroke();
    ctx.font = "bold 12px Courier";
    ctx.fillText("Box:" + id, xMin,yMin-2);
};

image.onload = function(e) {
    ctx.canvas.width = dimension;
    ctx.canvas.height = dimension;
    c.width = dimension;
    c.height = dimension;
    ctx.drawImage(image, 0, 0, dimension, dimension);
    console.log(`Box add ${boxes.length} ${boxes}`)
    for (i = 0; i < boxes.length; i++){
        drawBoxes(boxes[i].id, boxes[i].xMin, boxes[i].xMax, boxes[i].yMin, boxes[i].yMax);
    } 
};

image.style.display = "block";
image.src = "image/" + image_path;
var label = 0; 

var clicked = false;
var fPoint = {};
c.onclick = function(e) {
    console.log(clicked);
    if (!clicked) {
        var x = (dimension / c.scrollWidth) * e.offsetX;
        var y = (dimension / c.scrollHeight) * e.offsetY;
        console.log(e);
        ctx.strokeStyle = "red";
        ctx.fillStyle = "red";
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2*Math.PI, false);
        ctx.fill();
        fPoint = {
            x: x,
            y: y
        };
    } else {
        var x = (dimension / c.scrollWidth) * e.offsetX;
        var y = (dimension / c.scrollHeight) * e.offsetY;
        var xMin;
        var xMax;
        var yMin;
        var yMin;
        if (x > fPoint.x) {
            xMax = x;
            xMin = fPoint.x;
        } else {
            xMax = fPoint.x;
            xMin = x;
        }
        if (y > fPoint.y) {
            yMax = y;
            yMin = fPoint.y;
        } else {
            yMax = fPoint.y;
            yMin = y;
        }
        fPoint = {};
        console.log(`Box add`)
        window.location.replace("/add/" + (boxes.length + 1) +
        "?xMin=" + xMin +
        "&xMax=" + xMax +
        "&yMin=" + yMin +
        "&yMax=" + yMax);
        changed = 1;
    }
    clicked = !clicked;
};

console.log(`Key pressed ${color} \n Key code Value: ${label_text}`);

document.getElementById("demo").addEventListener("keypress", (event) => {
    changed = 0;
    var name = event.key;
    var code = event.code;
    console.log(`Key pressed ${name} \n Key code Value: ${code}`);
    const valid_keys = ['KeyP', 'KeyN', 'KeyU'];
    if (code === 'KeyP') {
        label = '1'; 
        changed = 1;
    }
    else if (code === 'KeyN') {
        label = '0'; 
        changed = 1;
    }
    else if (code === 'KeyU') {
        label = '-1'; 
        changed = 0;
        unsaved = false;
    }
    if (valid_keys.includes(code)) {
        window.location.replace("/classify/" + image + "?label=" + label);
        document.getElementById('labelSquare').style.backgroundColor = color;
        document.getElementById("labelText").innerHTML = label_text;
    }
}, false);

document.getElementById('labelSquare').style.backgroundColor = color;
document.getElementById("labelText").innerHTML = label_text;

document.getElementById("demo").addEventListener("keydown", (event) => {
    var name = event.key;
    var code = event.code;
    console.log(`Key pressed ${name} \n Key code Value: ${code}`);
    console.log(`Label ${changed} `);

    if (code === 'ArrowLeft') {
        if (changed === 1 && save_auto === 0) {
            unsaved = true
            window.dispatchEvent(new Event('beforeunload'))
        }
        window.location.replace("/prev");
        return;
    }
    else if (code === 'ArrowRight'){
        if (changed === 1 && save_auto === 0) {
            unsaved = true
            window.dispatchEvent(new Event('beforeunload'))
        }
        window.location.replace("/next");
        return;
    }
    else if (code === 'KeyS'){
        changed = 0;
        unsaved = false;
        window.location.replace("/save");
        return;
    }
}, false);

var timer

window.addEventListener('beforeunload', (event) => {
  timer = window.setTimeout( function () {
    unsaved = false;
  }, 50);
  // if (unsaved === true) {
  //   var confirmationMessage = 'Are you sure to leave the page?';  // a space
  //   (event || window.event).returnValue = confirmationMessage;
  //   return confirmationMessage;
  // }
});

window.addEventListener('beforeunload', (event) => {
  console.log(`UNSAVED ${unsaved} ${changed}`);
  if (unsaved === true) {
    event.preventDefault();
    event.returnValue = 'Are you sure to leave the page?';
  }
});

window.addEventListener("unload", function(event) {
  window.clearTimeout(timer)
})