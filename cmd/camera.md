
    To temporarily disable reverse path filtering for eth1, in the terminal, enter:

sudo sysctl -w net.ipv4.conf.all.rp_filter=0
sudo sysctl -w net.ipv4.conf.eth1.rp_filter=0

    To permanently disable reverse path filtering:

1. Comment out the lines specified below in /etc/sysctl.d/10-network-security:

# Turn on Source Address Verification in all interfaces to # in order to prevent some spoofing attacks.

## net.ipv4.conf.default.rp_filter=1

## net.ipv4.conf.all.rp_filter=1

2. Restart the computer.