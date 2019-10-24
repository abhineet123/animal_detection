<!-- MarkdownTOC -->

- [vot](#vot)
    - [p1_bear       @ vot](#p1_bear__vot)
        - [bear_1_1       @ p1_bear/vot](#bear11__p1_bearvot)
    - [p1_deer       @ vot](#p1_deer__vot)
    - [p1_coyote       @ vot](#p1_coyote__vot)
        - [coyote_jesse_static_1       @ p1_coyote/vot](#coyote_jesse_static_1__p1_coyotevot)
        - [coyote_jesse_static_2       @ p1_coyote/vot](#coyote_jesse_static_2__p1_coyotevot)
    - [p1_moose       @ vot](#p1_moose__vot)
- [vot18](#vot18)
    - [p1_bear       @ vot18](#p1_bear__vot18)
        - [bear_1_1       @ p1_bear/vot18](#bear11__p1_bearvot18)
    - [p1_deer       @ vot18](#p1_deer__vot18)
    - [p1_coyote       @ vot18](#p1_coyote__vot18)
        - [coyote_jesse_static_1       @ p1_coyote/vot18](#coyote_jesse_static_1__p1_coyotevot18)
        - [coyote_jesse_static_2       @ p1_coyote/vot18](#coyote_jesse_static_2__p1_coyotevot18)
    - [p1_moose       @ vot18](#p1_moose__vot18)
- [davis](#davis)
    - [p1_bear       @ davis](#p1_bear__davis)
        - [bear_1_1       @ p1_bear/davis](#bear11__p1_beardavis)
    - [p1_deer       @ davis](#p1_deer__davis)
    - [p1_coyote       @ davis](#p1_coyote__davis)
        - [coyote_jesse_static_1       @ p1_coyote/davis](#coyote_jesse_static_1__p1_coyotedavis)
        - [coyote_jesse_static_2       @ p1_coyote/davis](#coyote_jesse_static_2__p1_coyotedavis)
    - [p1_moose       @ davis](#p1_moose__davis)

<!-- /MarkdownTOC -->

<a id="vot"></a>
# vot

<a id="p1_bear__vot"></a>
## p1_bear       @ vot

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_bear.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask/bear --patch_tracker.siam_mask.config=config_vot.json --patch_tracker.siam_mask.resume=SiamMask_VOT.pth

<a id="bear11__p1_bearvot"></a>
### bear_1_1       @ p1_bear/vot

CUDA_VISIBLE_DEVICES=0
python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_path=/data/acamp/acamp20k/bear_1_1 --patch_tracker.siam_mask.config=config_vot.json --patch_tracker.siam_mask.resume=SiamMask_VOT.pth

<a id="p1_deer__vot"></a>
## p1_deer       @ vot

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_deer.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask/deer --patch_tracker.siam_mask.config=config_vot.json --patch_tracker.siam_mask.resume=SiamMask_VOT.pth

<a id="p1_coyote__vot"></a>
## p1_coyote       @ vot

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_coyote.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask/coyote --patch_tracker.siam_mask.config=config_vot.json --patch_tracker.siam_mask.resume=SiamMask_VOT.pth

<a id="coyote_jesse_static_1__p1_coyotevot"></a>
### coyote_jesse_static_1       @ p1_coyote/vot

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=coyote_jesse_static_1 --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask/coyote --patch_tracker.siam_mask.config=config_vot.json --patch_tracker.siam_mask.resume=SiamMask_VOT.pth --init_frame_id=0 --end_frame_id=0

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=coyote_jesse_static_1 --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask/coyote --patch_tracker.siam_mask.config=config_vot.json --patch_tracker.siam_mask.resume=SiamMask_VOT.pth --init_frame_id=1 --end_frame_id=1

<a id="coyote_jesse_static_2__p1_coyotevot"></a>
### coyote_jesse_static_2       @ p1_coyote/vot

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=coyote_jesse_static_2 --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask/coyote --patch_tracker.siam_mask.config=config_vot.json --patch_tracker.siam_mask.resume=SiamMask_VOT.pth --init_frame_id=1 --end_frame_id=1

<a id="p1_moose__vot"></a>
## p1_moose       @ vot

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_moose.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask/moose  --patch_tracker.siam_mask.config=config_vot.json --patch_tracker.siam_mask.resume=SiamMask_VOT.pth
 
<a id="vot18"></a>
# vot18

<a id="p1_bear__vot18"></a>
## p1_bear       @ vot18

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_bear.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_vot18/bear --patch_tracker.siam_mask.config=config_vot18.json --patch_tracker.siam_mask.resume=SiamMask_VOT_LD.pth

<a id="bear11__p1_bearvot18"></a>
### bear_1_1       @ p1_bear/vot18

CUDA_VISIBLE_DEVICES=0
python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_path=/data/acamp/acamp20k/bear_1_1 --patch_tracker.siam_mask.config=config_vot18.json --patch_tracker.siam_mask.resume=SiamMask_VOT_LD.pth

<a id="p1_deer__vot18"></a>
## p1_deer       @ vot18

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_deer.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_vot18/deer --patch_tracker.siam_mask.config=config_vot18.json --patch_tracker.siam_mask.resume=SiamMask_VOT_LD.pth

<a id="p1_coyote__vot18"></a>
## p1_coyote       @ vot18

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_coyote.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_vot18/coyote --patch_tracker.siam_mask.config=config_vot18.json --patch_tracker.siam_mask.resume=SiamMask_VOT_LD.pth

<a id="coyote_jesse_static_1__p1_coyotevot18"></a>
### coyote_jesse_static_1       @ p1_coyote/vot18

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=coyote_jesse_static_1 --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_vot18/coyote --patch_tracker.siam_mask.config=config_vot18.json --patch_tracker.siam_mask.resume=SiamMask_VOT_LD.pth --init_frame_id=0 --end_frame_id=0

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=coyote_jesse_static_1 --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_vot18/coyote --patch_tracker.siam_mask.config=config_vot18.json --patch_tracker.siam_mask.resume=SiamMask_VOT_LD.pth --init_frame_id=1 --end_frame_id=1

<a id="coyote_jesse_static_2__p1_coyotevot18"></a>
### coyote_jesse_static_2       @ p1_coyote/vot18

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=coyote_jesse_static_2 --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_vot18/coyote --patch_tracker.siam_mask.config=config_vot18.json --patch_tracker.siam_mask.resume=SiamMask_VOT_LD.pth --init_frame_id=1 --end_frame_id=1

<a id="p1_moose__vot18"></a>
## p1_moose       @ vot18

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_moose.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_vot18/moose  --patch_tracker.siam_mask.config=config_vot18.json --patch_tracker.siam_mask.resume=SiamMask_VOT_LD.pth
 
<a id="davis"></a>
# davis

<a id="p1_bear__davis"></a>
## p1_bear       @ davis

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_bear.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_davis/bear --patch_tracker.siam_mask.config=config_davis.json --patch_tracker.siam_mask.resume=SiamMask_DAVIS.pth

<a id="bear11__p1_beardavis"></a>
### bear_1_1       @ p1_bear/davis

CUDA_VISIBLE_DEVICES=0
python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_path=/data/acamp/acamp20k/bear_1_1 --patch_tracker.siam_mask.config=config_davis.json --patch_tracker.siam_mask.resume=SiamMask_DAVIS.pth

<a id="p1_deer__davis"></a>
## p1_deer       @ davis

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_deer.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_davis/deer --patch_tracker.siam_mask.config=config_davis.json --patch_tracker.siam_mask.resume=SiamMask_DAVIS.pth

<a id="p1_coyote__davis"></a>
## p1_coyote       @ davis

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_coyote.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_davis/coyote --patch_tracker.siam_mask.config=config_davis.json --patch_tracker.siam_mask.resume=SiamMask_DAVIS.pth

<a id="coyote_jesse_static_1__p1_coyotedavis"></a>
### coyote_jesse_static_1       @ p1_coyote/davis

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=coyote_jesse_static_1 --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_davis/coyote --patch_tracker.siam_mask.config=config_davis.json --patch_tracker.siam_mask.resume=SiamMask_DAVIS.pth --init_frame_id=0 --end_frame_id=0

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=coyote_jesse_static_1 --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_davis/coyote --patch_tracker.siam_mask.config=config_davis.json --patch_tracker.siam_mask.resume=SiamMask_DAVIS.pth --init_frame_id=1 --end_frame_id=1

<a id="coyote_jesse_static_2__p1_coyotedavis"></a>
### coyote_jesse_static_2       @ p1_coyote/davis

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=coyote_jesse_static_2 --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_davis/coyote --patch_tracker.siam_mask.config=config_davis.json --patch_tracker.siam_mask.resume=SiamMask_DAVIS.pth --init_frame_id=1 --end_frame_id=1

<a id="p1_moose__davis"></a>
## p1_moose       @ davis

python3 tracking/Server.py --mode=2 --patch_tracker.tracker_type=3 --img_paths=p1_moose.txt --root_dir=/data/acamp/acamp20k/prototype_1_source --save_dir=siam_mask_davis/moose  --patch_tracker.siam_mask.config=config_davis.json --patch_tracker.siam_mask.resume=SiamMask_DAVIS.pth
 

