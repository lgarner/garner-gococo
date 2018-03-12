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
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"golang.org/x/image/colornames"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"net/http"
	"io"
	"crypto/tls"
	"strconv"
)

// Global labels array
var labels []string

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

// TENSOR UTILITY FUNCTIONS
func makeTensorFromImage(filename string) (*tf.Tensor, image.Image, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, err
	}

	r := bytes.NewReader(b)
	img, _, err := image.Decode(r)

	if err != nil {
		return nil, nil, err
	}

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		return nil, nil, err
	}
	// Creates a tensorflow graph to decode the jpeg image
	graph, input, output, err := decodeJpegGraph()
	if err != nil {
		return nil, nil, err
	}
	// Execute that graph to decode this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, nil, err
	}
	return normalized[0], img, nil
}

func decodeJpegGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func loadLabels(labelsFile string) {
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
}

func getLabel(idx int, probabilities []float32, classes []float32) string {
	index := int(classes[idx])
	label := fmt.Sprintf("%s (%2.0f%%)", labels[index], probabilities[idx]*100.0)

	return label
}

func addLabel(img *image.RGBA, x, y, class int, label string) {
	col := colornames.Map[colornames.Names[class]]
	point := fixed.Point26_6{fixed.Int26_6(x * 64), fixed.Int26_6(y * 64)}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(colornames.Black),
		Face: basicfont.Face7x13,
		Dot:  point,
	}

	Rect(img, x, y-13, (x + len(label)*7), y-6, 7, col)

	d.DrawString(label)
}

var psession *tf.Session
var pgraph *tf.Graph
var outjpg string = "output.jpg"


func download(URL, filename string) error {
	// Not pretty, but we have a self signed cert.
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	client := &http.Client{Transport: tr}

	resp, err := client.Get(URL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer file.Close()
	_, err = io.Copy(file, resp.Body)
	return err
}

func runClassifier(jpgfile string, outjpg string) {
	graph := pgraph
	log.Println("jpgfile: ", jpgfile)
	log.Println("outjpg: ", outjpg)
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, i, err := makeTensorFromImage(jpgfile)
	if err != nil {
		log.Fatal(err)
	}

	// Transform the decoded YCbCr JPG image into RGBA
	b := i.Bounds()
	img := image.NewRGBA(b)
	draw.Draw(img, b, i, b.Min, draw.Src)

	// Get all the input and output operations
	inputop := graph.Operation("image_tensor")
	// Output ops
	o1 := graph.Operation("detection_boxes")
	o2 := graph.Operation("detection_scores")
	o3 := graph.Operation("detection_classes")
	o4 := graph.Operation("num_detections")

	// Execute COCO Graph
	output, err := psession.Run(
		map[tf.Output]*tf.Tensor{
			inputop.Output(0): tensor,
		},
		[]tf.Output{
			o1.Output(0),
			o2.Output(0),
			o3.Output(0),
			o4.Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}

	// Outputs
	probabilities := output[1].Value().([][]float32)[0]
	classes := output[2].Value().([][]float32)[0]
	boxes := output[0].Value().([][][]float32)[0]

	// Draw a box around the objects
	curObj := 0

	// 0.4 is an arbitrary threshold, below this the results get a bit random
	for probabilities[curObj] > 0.4 {
		x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
		x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]

		Rect(img, int(x1), int(y1), int(x2), int(y2), 4, colornames.Map[colornames.Names[int(classes[curObj])]])
		addLabel(img, int(x1), int(y1), int(classes[curObj]), getLabel(curObj, probabilities, classes))

		curObj++
	}

	// Output JPG file
	outfile, err := os.Create(outjpg)
	if err != nil {
		log.Fatal(err)
	}

	var opt jpeg.Options

	opt.Quality = 80

	err = jpeg.Encode(outfile, img, &opt)
	if err != nil {
		log.Fatal(err)
	}
}

var MAX_CLIENTS int
var urlClassifyChannel chan struct{}

func bytehandler(w http.ResponseWriter, r *http.Request) {
	log.Println("In Classifier.")
	var url string

	if (r.Method == "POST") {
		r.ParseForm()
		// The file to classify:
		log.Println("Classify this: ")
		url = r.Form.Get("filename") // yes, not safe.
		log.Println(r.Form.Get("filename"))

		// FIXME: Retrieve file, should just copy request body bytes directy to tensorflow input.
		err := download(url, "tmpFile.jpg") // Session key would be nice to prepend.
		if err != nil {
			return
		}
	} else {
		return
	}
	// Block until channel frees (via defer). Seems strange, but it's a counting semaphore.
	urlClassifyChannel <- struct{}{}
	defer func() { <-urlClassifyChannel }()

	// Run classifier, which writes to ouptut.jpg
	runClassifier("tmpFile.jpg", "output.jpg")


	// Open output.
	var reader io.Reader
	var err error
	reader, err = os.Open("output.jpg")
	b, err := ioutil.ReadAll(reader) // Hmm. Should IO stream it.
	if (err != nil) {
		log.Println("In Classifier. No output!")
		return // Nothing to do.
	}
	// Write byte output.


	//w.Header().Set("Content-Type", "image/jpeg")
	log.Println("Size of file: %d", len(b))
	//w.Header().Set("Content-Length", strconv.Itoa(len(b)))
	if _, err := w.Write(b); err != nil {
		log.Println("unable to write image.")
	}
	log.Println("In Classifier. Written!")

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

	// Https seems to be needed for pusher by browser standard.
	log.Fatal(http.ListenAndServeTLS(*httpAddr, "certs/cert.pem", "certs/key.pem", nil))
}

func main() {
	// Parse flags
	var err error
	modeldir := flag.String("dir", "", "Directory containing COCO trained model files. Assumes model file is called frozen_inference_graph.pb")
	jpgfile := flag.String("jpg", "", "Path of a JPG image to use for input")
	labelfile := flag.String("labels", "labels.txt", "Path to file of COCO labels, one per line")
	maxRequests := flag.String("maxrequests", "100", "Max number of requests")
	MAX_CLIENTS, err = strconv.Atoi(*maxRequests)
	if (MAX_CLIENTS < 1 || err != nil) {
		flag.Usage()
		return
	}
	flag.Parse()
	if *modeldir == "" || *jpgfile == "" {
		flag.Usage()
		return
	}

	// Load the labels
	loadLabels(*labelfile)
	urlClassifyChannel = make(chan struct{}, MAX_CLIENTS)

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

	//runClassifier(*jpgfile, "output.jpg")
	launchHTTPListeners()

}


const indexHTML = `<html>
<head>
	<title>Hello World</title>
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