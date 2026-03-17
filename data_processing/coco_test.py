from cocoapi.PythonAPI.pycocotools.coco import COCO
import os
import cv2

from eval_utils import  drawBox


dataDir = 'N:\Datasets\COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cat_names)))

sup_cat = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(sup_cat)))

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
_pause = 1
for img_id in imgIds:
    img = coco.loadImgs(img_id)[0]

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    img_np = cv2.imread(os.path.join(dataDir, dataType, img['file_name']))

    # print('img: ', img)
    # print('anns: ', anns)

    for ann in anns:
        print('ann: ', ann)

        bbox = ann['bbox']
        category_id = ann['category_id']

        label = cat_names[category_id - 1]


        bbox = [int(x) for x in bbox]
        x, y, w, h = bbox

        xmin = x
        ymin = y

        xmax = xmin + w
        ymax = ymin + h

        box_color = (0, 255, 0)
        drawBox(img_np, xmin, ymin, xmax, ymax, box_color, label)


    cv2.imshow('img_np', img_np)
    wait_dur = 500 if not _pause else 0

    k = cv2.waitKey(wait_dur) & 0xFF
    if k == ord('q') or k == 27:
        break
    elif k == 32:
        _pause = 1- _pause


