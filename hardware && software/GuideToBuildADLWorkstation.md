# This is a guide writen in Feb 2020 about how to build up a small deep learning workstation.

#### Small Comments

I put the link of every item on Amazon basically because I want the link to be valid for as long as possible, so that pissible readers may at least have a look of the item. But the price on Amazon may not always be feasible. Ebay, Neweggs are also very good platforms with lower prices.
## Hardware List
* CPU: Intel i9-9820x 10 core 20 thread. [Amazon](https://www.amazon.ca/Intel-i9-9820X-Processor-Unlocked-Processors/dp/B07KCCH7JL/ref=sr_1_1?keywords=9820x&qid=1581821067&sr=8-1)
  
  Genrally, this building tends to use Intel CPU for faster speed per thread, but it's also true that AMD CPU provides way more cores and PCIE channels if averaged on their prices. In 2020, AMD also released brand new threadripper CPUs, which are also highly worth consideration when building deep learning machines in future. Specially, most of the AMD threadripper provide more than 64 PCIE channels, which can definitely meet the requirement of 4 GPUs (16 channels per GPU to achieve full data transfer speed).

* GPU: ASUS RTX 2080ti 11Gb GDDR6 turbo. [Amazon](https://www.amazon.ca/GeForce%C2%AE-Turbo-Type-C-Graphics-TURBO-RTX2080TI-11G/dp/B07GK2LWDL/ref=sr_1_1?keywords=2080ti+turbo&qid=1581822891&sr=8-1)
  
  For GPU, while actually only Nvidia GPUs are pratically possible to run common deep learning libraries like pytorch, tensorflow... It's not exactly impossible to do deep learning with AMD GPUs, but it will require dealing with a lot more low level problems.

  If you are reading this far later than 2020, maybe 2080ti is not suitable for you. The most economic choice for a researcher is always the newest type in Nvidia RTX/GTX family. Never try Titan ot Tesla, since they are actually not designed for common use and have a much lower price-quality ratio.

  Also, if you are building a multi-gpu workstation, then you'd better choose turbo coolingh GPU. Fan-cooling will be good for a single GPU but pretty bad for cooling multiple GPUs. You will understand this if you actually see how a motherboard looks like, usually there is less than 5mm between 2 neighboring GPUs. Fan-cooling will always make one GPU take all heat from the other beneath it. Another choice is [Gigabyte 2080ti turbo](https://www.gigabyte.com/ca/Graphics-Card/GV-N208TTURBO-11GC-rev-10#kf).

  Besides, the cost on GPU should be over 60% of the total. But typically the GPU preferred by deep learning researchers are way more expensive than the common choice of computer gamers. So the price of these GPU may vary a lot, because very few customers will choose it. And it's highly recommended to look around in your local computer component shops before purchase it online. Usually the prices in physical shops are even much lower than that on Amazon. For example, I purchased the ASUS 2080ti turbo in [Mike's Computer Shop](https://mikescomputershop.com/) for around 1300 CAD, while on Amazon the price is now higher than 2000 CAD.

* Motherboard: Gigabyte X299 Auros master [Amazon](https://www.amazon.ca/GIGABYTE-X299-AORUS-Master-Motherboards/dp/B07KZGRCV3/ref=sr_1_5?keywords=auros+master&qid=1581823394&sr=8-5)
  
  Just make sure your mother board mathches your CPU, then everything is okay. There is an [Asus WS X299 SAGE](https://www.amazon.ca/Asus-X299-SAGE-Workstation-Motherboard/dp/B07GKZ5NRB/ref=sr_1_1?keywords=motherboard+x299+sage&qid=1581823480&sr=8-1) which can provide 64 PCIE channels while LGA2066 CPU only support 44 PCIE in total, using some on-board fancy technology. But the price is also much higher. 
* Memory: Corsair Vegnance LPX 64GB(4*16 GB) 2133 Hz[Amazon](https://www.amazon.ca/Corsair-Vengeance-288-Pin-Memory-CMK32GX4M2B3200C16/dp/B0196QNBU4/ref=sr_1_3?keywords=corsair%2Bvengeance%2Blpx%2B32gb&qid=1581825095&sr=8-3&th=1)
  
  Frequecy doesn't make a lot difference when doing deep learning, but the size really do. At least make sure there are (20GB * number of your GPUs) memory available.

* Internal SSD: Samsung Evo Plus 500GB [Amazon](https://www.amazon.ca/Samsung-970-EVO-Plus-MZ-V7S500B/dp/B07M7Q21N7/ref=sr_1_1?keywords=samsung+evo+plus&qid=1581825281&sr=8-1)
  
  The motherboard at least support 8 SATA peripheral disks, so no need to buy a too big internal SSD.
* CPU Cooler: CORSAIR HYDRO SERIES H115i PRO [Amazon](https://www.amazon.ca/CORSAIR-Radiator-Advanced-Lighting-Software/dp/B077G3C6HH/ref=sr_1_1_sspa?keywords=corsair+cooler+h115i+pro&qid=1581827696&sr=8-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzOTJQT1FMR1VHUk8wJmVuY3J5cHRlZElkPUEwMDMwMjk0MUFWNExBOVVKQ1ZGNyZlbmNyeXB0ZWRBZElkPUEwODUxNjg3MlZPVjhPWFQ1SDlFOCZ3aWRnZXROYW1lPXNwX2F0ZiZhY3Rpb249Y2xpY2tSZWRpcmVjdCZkb05vdExvZ0NsaWNrPXRydWU=)
  
  Personally I don't like RGB, so maybe I will shoose [CORSAIR HYDRO SERIES H110i](https://www.amazon.ca/CORSAIR-Radiator-Advanced-Lighting-Software/dp/B019955W7C/ref=sr_1_1_sspa?keywords=corsair%2Bcooler%2Bh115i%2Bpro&qid=1581827696&sr=8-1-spons&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzOTJQT1FMR1VHUk8wJmVuY3J5cHRlZElkPUEwMDMwMjk0MUFWNExBOVVKQ1ZGNyZlbmNyeXB0ZWRBZElkPUEwODUxNjg3MlZPVjhPWFQ1SDlFOCZ3aWRnZXROYW1lPXNwX2F0ZiZhY3Rpb249Y2xpY2tSZWRpcmVjdCZkb05vdExvZ0NsaWNrPXRydWU&th=1) if I rebuild it.

* Case: Corsair Carbide Series Air 540 High Airflow ATX Cube Case CC-9011030-WW - Black [Amazon](https://www.amazon.ca/Corsair-Carbide-High-Airflow-CC-9011030-WW/dp/B00D6GINF4/ref=sr_1_8?crid=28BM1OFBFX3CN&keywords=corsair+atx+case&qid=1581829077&sprefix=corsair+ATX%2Caps%2C205&sr=8-8)
  
  Not every case is able to hold 4 GPU and the 1600w PSU. This case is also very good for cooling, and easy for arranmging wires. Besdes, this case also has plenty of place to hold many peripheral hard disks, enough for usual laboratory use. Besides, not every case provides snough space for a 280mm CPU water cooler.
* PSU: ROSEWILL Gaming 80 Plus Gold 1600W Power Supply [Amazon](https://www.amazon.ca/ROSEWILL-HERCULES-Certified-Extra-long-CrossFire/dp/B00PCLGZOC/ref=sr_1_7?keywords=power+1600w&qid=1581829211&sr=8-7)
  
  1600w is necessary for 4 GPUs. 1500w will cause the motherboard shutdown accidentally due to power undersupply.

## Assembly Guide
This guide tries to gather necessary information for assembly every part of a PC, many people provide a lot of detailed videos on Youtube about this, so this is an easy task. It's recommended to install all the things following the ordor listed below.
* [Put CPU onto the motherboard.](https://www.youtube.com/watch?v=_zojIW-2DD8)
* [Install the memory stick.](https://www.youtube.com/watch?v=v3J9VtWMEE8) 
* [Install the motherboard into the case.](https://www.youtube.com/watch?v=oI4TZGLxy78)
* [Install the CPU cooler.](https://www.youtube.com/watch?v=UWt22EV8m3Q)
* [Install the GPU](https://www.youtube.com/watch?v=HoLv2s23mMQ)
* [Arrange all the wires](https://www.youtube.com/watch?v=UixqA7Exk_I)
  
  You need some patience on this, make sure read your motherboard mannual, PSU mannual and case mannual before doing this. For example, it took me a long time to realize SATA power is different from SATA, the former is on PSU while the latter is on motherboard.
* Power on. Hopefully you will succeed all at once, but that doesn't usually happen. You need to look for professional experts if you cannot figure out which wire is connected to a wrong place.  

## Installation Guide
* It's generally not recommended to download a very new version of [Ubuntu server](https://ubuntu.com/download/server). For me, I found the Ubuntu server 18.04 now turns to a 'live' installler, causing installation failure. I use the 16.04 LTS.

* Install the Nvidia GPU driver. You don't really need the newest driver to do deep learning, actually most of the driver updates are for better gaming experience. My driver version is 418.87.00, while the latest one should 430.xx.xx If youy are not sure what version is suitable, open a [Google Colab](https://colab.research.google.com/), use the command 

  ```python
  !nvidia-smi
  ```
  to check what version Google uses. But normally, following this [tutorial](https://www.linuxbabe.com/ubuntu/install-nvidia-driver-ubuntu-18-04) is enough. 
* Install CUDA. This is necessary for every deep learning frame like Pytorch. The [official cuda doc](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) lists several ways to do this in detail.
* Install cudnn, following this [official doc](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

This is a long and complex way to install everything, while perhaps the newest cuda version you just installed is not supported by Pytorch or Tensorflow. An much easier way is just to install pytorch, cuda all together by conda or pip, with this command which you should be able to find at the official PyTorch website.
```shell
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```  
It turns out that when intalling cuda, it will also install a linux-friendly Nvidia driver automatically.

* For multi-user situation, it's recommened to install a [Docker Engine](https://docs.docker.com/install/linux/docker-ce/ubuntu/) on the Linux server, because it virtualizes the Operating System so that now no user can even break the OS in a theoretically possible way. It also isolates the OS from virus amd worm attack.
* Another problem met by most of Ubuntu16.04 user is that the defualt python version is python 2.7, even explicitly installed python3 is only 3.5, which is not new enough for most of the popular development -- python >= 3.6

  So it's very important to change the default python interpreter version of this system. [Stackoverflow](https://stackoverflow.com/questions/43621584/why-cant-i-install-python3-6-dev-on-ubuntu16-04)

  Meanwhile, make sure your default 'pip install' is pointing to the new python3.6 interpreter. Very possibly it's not, you need to find a way to change it.

  Also, other necessary python develop tools like 'sudo apt install python3-dev' now should be 'sudo apt install python3.6-dev'.

  Well, after doing this, perhaps your termal won't launch. It's quite annoying that the terminal (actually terminal simulator) nowadays actually depends on python to run, if you are working under a desktop environment but not a real server. To solve this, you may have to revert back to your system default python interpreter. To do this, you may have to understand how to launch a real terminal in a desktop environment, not the usual simulated terminal.
    
