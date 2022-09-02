var c = document.getElementById("canvas");
var ctx = c.getContext("2d");
var image = new Image();
console.log(image);
image.onload = function(e) {
    ctx.drawImage(image, 0, 0, 256, 256);
};
image.style.display = "block";
image.src = "image/" + image_path;
var label = 0; 
console.log(`Key pressed ${color} \n Key code Value: ${label_text}`);

document.getElementById("demo").addEventListener("keypress", (event) => {
    var name = event.key;
    var code = event.code;
    console.log(`Key pressed ${name} \n Key code Value: ${code}`);

    if (code === 'KeyP') {
        label = 1; 
    }
    else if (code === 'KeyN') {
        label = 0; 
    }
    if (code === 'KeyP' || code === 'KeyN') {
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

    if (code === 'ArrowLeft') {
        window.location.replace("/prev");
        return;
    }
    else if (code === 'ArrowRight'){
        window.location.replace("/next");
        return;
    }
}, false);
