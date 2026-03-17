# install

sudo add-apt-repository ppa:mc3man/trusty-media

sudo apt-get update

sudo apt-get install ffmpeg

# local streaming

## encode and stream

ffmpeg -r 30 -f v4l2 -i /dev/video0 -pix_fmt yuv420p -an -preset x264 -f mpegts udp://127.0.0.1:8888

ffmpeg -r 30 -f v4l2 -i /dev/video0 -pix_fmt yuv420p -an -c:v libx264 -preset slow -crf 0 -f mpegts udp://127.0.0.1:8888

## decode and playback

ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental udp://127.0.0.1:8888

# remote streaming

## encode and stream

ffmpeg -f v4l2 -i /dev/video0 -pix_fmt yuv420p -an -preset x264 -f mpegts udp://192.168.0.132:8888

### desktop

ffmpeg -f x11grab  -r 30 -s 1280x720 -i :0 -pix_fmt yuv420p -an -preset x264 -f mpegts udp://192.168.0.132:8888


## decode and playback

ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental udp://192.168.0.132:22

# miscellaneous commands

v4l2-ctl --list-devices

ffmpeg -f v4l2 -list_formats all -i /dev/video0

ffmpeg -f v4l2 -framerate 25 -video_size 640x480 -i /dev/video0 output.mkv