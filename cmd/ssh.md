(awk '{print $2}' <(ifconfig enp0s3 | grep 'inet ')); 
enp0s8

auto enp0s8
iface enp0s8 inet static
    address 192.168.56.101
    netmask 255.255.255.0