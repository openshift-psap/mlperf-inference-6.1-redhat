# Container set up and running the haness with gpt-oss-120b 

## Setup the container 

### Pull the image 

```bash
 podman  pull vllm/vllm-openai:v0.15.0-cu130
```

###  Run the container 
```python
 podman run -d --pids-limit=32768  --name vllm-pod-0.15.0-cu130 --device nvidia.com/gpu=all    --security-opt=label=disable  --net host 
-e HF_HOME   -v /mnt/data/:/mnt/data/ --ipc=host --entrypoint /bin/bash  vllm/vllm-openai:v0.15.0-cu130  -lc 'tail -f /dev/null'
```

### Exec into the container 

```bash
podman exec -it vllm-pod-0.15.0-cu130  /bin/bash
```

### Check the pid limit 
```bash
 cat /sys/fs/cgroup/pids.max
```

### Set the correct libcuda.so path in the container 
```python
#Check where is the current libcuda pointed to 
grep -R "/usr/local/cuda-13.0/compat" /etc/ld.so.conf.d

#Should show the file which it is currently pointing to 
/etc/ld.so.conf.d/00-cuda-compat.conf:/usr/local/cuda-13.0/compat/

#Disable this config file 
mv /etc/ld.so.conf.d/00-cuda-compat.conf /etc/ld.so.conf.d/00-cuda-compat.conf.disabled && ldconfig

#Check the output
python3 -c "import torch; print(torch.cuda.is_available())"
```


