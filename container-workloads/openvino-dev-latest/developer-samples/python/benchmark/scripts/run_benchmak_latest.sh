tag=2022.2
#sudo docker run  --env-file conftest.env benchmarking_2021.4:latest
sudo docker run -e RUN_ON_PREM=data  benchmarking_$tag:latest
#sudo docker run  --env-file conftest1.env benchmark_2021.3:latest
