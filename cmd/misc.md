# nvidia_e5g

sudo mv /var/lib/dpkg/info/nvidia-384-dev.* /tmp/
sudo dpkg --remove --force-remove-reinstreq nvidia-384-dev
sudo apt-get remove nvidia-384-dev

sudo mv /var/lib/dpkg/info/nvidia-331-dev.* /tmp/
sudo dpkg --remove --force-remove-reinstreq nvidia-331-dev
sudo apt-get remove nvidia-331-dev


sudo apt-get autoremove && sudo apt-get autoclean

# acamp20k

mv coyote_doc_1_1 _new/
mv coyote_doc_5_1 _new/
mv coyote_doc_9_1 _new/
mv coyote_doc_11_1 _new/
mv coyote_doc_13_1 _new/
mv coyote_doc_27_1 _new/



# count_files

python2 /home/abhineet/PTF/countFileInSubfolders.py jpg list.txt /data/acamp/acamp5k/train/images

python2 /home/abhineet/H/UofA/MSc/Code/TrackingFramework/countFileInSubfolders.py file_ext=jpg out_file=list.txt folder_name=/home/abhineet/N/Datasets/Acamp/acamp5k/train/images/

python2 /home/abhineet/H/UofA/MSc/Code/TrackingFramework/countFileInSubfolders.py file_ext=jpg out_file=list.txt folder_name=/home/abhineet/N/Datasets/Acamp/acamp5k/test/images/


# moveSubfolders

python3 moveSubfolders.py src_dir=/data/acamp/acamp10k/June_22_Annotations dst_dir=/data/acamp/acamp10k

python3 moveSubfolders.py src_dir=/data/acamp/acamp10k/June_25_Annotations dst_dir=/data/acamp/acamp10k

python3 moveSubfolders.py src_dir=/data/acamp/acamp20k/bear dst_dir=/data/acamp/acamp10k

python3 moveSubfolders.py src_dir=/data/acamp/acamp20k/moose dst_dir=/data/acamp/acamp20k/moose