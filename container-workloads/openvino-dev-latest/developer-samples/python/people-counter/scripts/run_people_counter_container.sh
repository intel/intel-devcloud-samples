tag=2022.1
# run on local machine
sudo docker run  -e RUN_ON_PREM=data -it people-counter_$tag:latest
# run on on prem cluster
#sudo docker run   -it object_detection_$tag_fp16:latest
