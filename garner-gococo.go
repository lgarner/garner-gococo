/*
gococo - CLI for executing COCO TensorFlow graphs

Copyright (c) 2017 ActiveState Software

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"path/filepath"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"net/http"
	"github.com/lgarner/workerpool"
	"strconv"
)



// DRAWING UTILITY FUNCTIONS

// HLine draws a horizontal line
func HLine(img *image.RGBA, x1, y, x2 int, col color.Color) {
	for ; x1 <= x2; x1++ {
		img.Set(x1, y, col)
	}
}

// VLine draws a veritcal line
func VLine(img *image.RGBA, x, y1, y2 int, col color.Color) {
	for ; y1 <= y2; y1++ {
		img.Set(x, y1, col)
	}
}

// Rect draws a rectangle utilizing HLine() and VLine()
func Rect(img *image.RGBA, x1, y1, x2, y2, width int, col color.Color) {
	for i := 0; i < width; i++ {
		HLine(img, x1, y1+i, x2, col)
		HLine(img, x1, y2+i, x2, col)
		VLine(img, x1+i, y1, y2, col)
		VLine(img, x2+i, y1, y2, col)
	}
}

var psession *tf.Session
var pgraph *tf.Graph
var classifier *workerpool.Classifier
var labels []string
var outjpg string = "output.jpg"
var dispatcher workerpool.Dispatcher

func bytehandler(w http.ResponseWriter, r *http.Request) {
	log.Println("In Classifier.")
	var url string
	if (r.Method == "POST") {
		r.ParseForm()
		// The file to classify:
		log.Println("Classify this: ")
		url = r.Form.Get("filename") // yes, not safe.
		log.Println(r.Form.Get("filename"))
	} else {
		return
	}

	payload := workerpool.NewPayload(url, "output.jpg", w, classifier)
	work := workerpool.Job{Payload: payload}

	// Push the work onto the queue.
	dispatcher.JobQueue <- work

	w.WriteHeader(http.StatusOK)
}

var httpAddr = flag.String("http", ":8081", "Listen address")
func launchHTTPListeners() {

	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { // Sorta dangerous with (fake) certs here.
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		pusher, ok := w.(http.Pusher)
		if ok {

			// Push is supported. Try pushing rather than
			// waiting for the browser request these static assets.
			if err := pusher.Push("/static", nil); err != nil {
				log.Printf("Failed to push: %v", err)
			}
			if err := pusher.Push("/static/style.css", nil); err != nil {
				log.Printf("Failed to push: %v", err)
			}
		}
		fmt.Fprintf(w, indexHTML)
	})

	// TODO: Grab any client given image from web, then just post as a webservice.
	http.HandleFunc("/dyn/classify", bytehandler)


	// Start the dispatcher, which starts the workers:
	dispatcher = *workerpool.NewDispatcher(MAX_WORKERS, MAX_QUEUE)
	dispatcher.Run()

	// Blocks.
	// Https seems to be needed for pusher by browser standard.
	log.Fatal(http.ListenAndServeTLS(*httpAddr, "certs/cert.pem", "certs/key.pem", nil))
}

var MAX_WORKERS int
var MAX_QUEUE int
func main() {
	// Parse flags
	var err error
	modeldir := flag.String("dir", "", "Directory containing COCO trained model files. Assumes model file is called frozen_inference_graph.pb")
	jpgfile := flag.String("jpg", "", "Path of a JPG image to use for input")
	labelfile := flag.String("labels", "labels.txt", "Path to file of COCO labels, one per line")
	maxWorkers := flag.String("maxworkers", "100", "Max workers to limit mem usage of unlimited go routines classifiers.")
	maxQueue := flag.String("maxqueue", "100", "Max job queue. Probably bigger than max workers")
	MAX_WORKERS, err = strconv.Atoi(*maxWorkers)
	MAX_QUEUE, err = strconv.Atoi(*maxQueue)
	flag.Parse()
	if *modeldir == "" || *jpgfile == "" {
		flag.Usage()
		return
	}

	// Load a frozen graph to use for queries

	modelpath := filepath.Join(*modeldir, "frozen_inference_graph.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	pgraph = graph // Keep a live pointer to reuse.
	if err := pgraph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	psession = session // Keep a live pointer to reuse.
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// Load the labels
	classifier = workerpool.NewClassifier(pgraph, psession, *labelfile)

	launchHTTPListeners()

}


const indexHTML = `<html>
<head>
	<title>Garner's Gococo Fork</title>
	<script src="/static/app.js"></script>
	<link rel="stylesheet" href="/static/style.css"">
</head>
<body onload = "startTimer()">
	<form>
    <input id="urlsrc" type="text" name="imgurl" value="">
    <button type="button" onclick="submitForClassificationUrl('urlsrc', 'classifiedimg')">Classify This Image</button>
	</form>
    <button type="button" onclick="submitForClassification('srcimg', 'classifiedimg')">Classify This Image</button>
	<div></div>
    <button type="button" onclick="displayPreviousImage()">Previous</button>
    <button type="button" onclick="displayNextImage()">Next</button>
	<div id="htmlimg">
    	<img id="srcimg" src="/static/pexels-photo-179124.jpeg"/>
    	<img id="classifiedimg" src="/output.jpg"/>
	</div>
    

</body>

</html>
`