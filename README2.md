Very quick notes:

Golang is based on workspaces, and the garner-gococo is just another project
in that workspace. Create a go directory, and update BASHRC (ex: export GOPATH=somwehere/go):

Get code:
mkdir -p $HOME/go/github.com/lgarner
cd $HOME/go/github.com/lgarner
git clone https://github.com/lgarner/garner-gococo.git
cd garner-gococo

Install "dep":
https://github.com/golang/dep

dep init
dep ensure
go build garner-gococo.go

This populates the go workspace with parallel project dependencies, and builds. If you can't install dep, go get <missing package URL path>. It drops
it into your workspace go/src directory.

Grab tensorflow models and extract it to the project root. You can read the
original gococo README.md to get that instructions (basically download a bunch
to play with).

That one points to some Tensorflow Model Zoo commit #:
https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md

Launch server (from the correct directory). "Goland" can be used to debug/set
breakpoints.

If the graph is extracted to the root, and it is ssd_mobile_net_v1_coco_11_06_2017, then you can type:

go run garner-gococo.go -dir ssd_mobilenet_v1_coco_11_06_2017 -http localhost:8082

Point browser to that https location with firefox, and add a site exception for
the self signed certificate so you can visit the page. Other browsers may be
used, but you'll need to allow that fake SSL certificate.

On that page, you have 3 images, or an arbitrary URL to play with the
garner-gococo's project's web front end to the selected tensorflow model.


