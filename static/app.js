console.log("Hello!");

function displayNextImage() {
  x = (x === images.length - 1) ? 0 : x + 1;
  document.getElementById("srcimg").src = images[x];
}

function displayPreviousImage() {
  x = (x <= 0) ? images.length - 1 : x - 1;
  document.getElementById("srcimg").src = images[x];
}

function startTimer() {
    setInterval(displayNextImage, 60000);
}

var images = [], x = 0;
images[0] = "/static/pexels-photo-179124.jpeg"
images[1] = "/static/pexels-photo.jpg"
images[2] = "/static/stock-photo-nature-nobody-outdoors-wildlife-water-sunlight-cloud-sky-silhouette-e88efff4-f7f6-4660-815e-0702ffc30a54.jpg"


function submitForClassification(idsrc, idTarget) {
    console.log("submitForClassification!");

    imgStr = document.getElementById(idsrc).src
    if (imgStr === null) {
        return
    }
    // HTTP GET message to server. An async message would be nice.
    var xhr = new XMLHttpRequest();
    // Async --> true. Server writes output.jpg to Writer directly, closes, and innerHTML gets updated.
    xhr.open('POST', "dyn/classify", true);

    xhr.onreadystatechange = function() {
        //console.log("status: " + this.status);
        //console.log("readyState: " + this.readyState);
        // 1 = Opened, 2 = headers, 3 = loading. 4 == DONE state. Can now read buffers.
        if (this.readyState == 4 && this.status == 200) {
            arrayBufferView = new Uint8Array(xhr.response),
                blob = new Blob([arrayBufferView], {'type': 'image\/jpeg'}),
                objectURL = window.URL.createObjectURL(blob);
            document.getElementById(idTarget).src = objectURL; // Binary assign Image to src.
        }
    };

    xhr.responseType = 'arraybuffer';
    xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhr.send("filename="+encodeURI(imgStr));

}
