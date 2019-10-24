import os, sys, glob, re
import numpy as np
import pandas as pd
import cv2
import inspect
from pprint import pprint
from PIL import Image

sys.path.append('..')
from tf_api.utilities import processArguments, sortKey, resizeAR, map_mask_to_bbox
from tracking.Visualizer import ImageWriter
from libs.shape import Shape
import tracking.Utilities as utils


def fixMasks(img_path, mask_path, csv_path, out_size='', out_dir='',
             save_video=1, write_text=0, show_img=1, frames_reader=None,
             img_ext='jpg', mask_ext='png', out_ext='mkv', codec='H264', fps=30, border=0, fixed_ar=0,
             include_binary=1, include_orig=1, show_bbox=1, map_to_bbox=0, apply_contour=0,
             writer=None, raw_mask=0):
    global _pause, _exit

    if map_to_bbox and not csv_path:
        csv_path = os.path.join(img_path, 'annotations.csv')

    img_files = [os.path.join(img_path, k) for k in os.listdir(img_path) if
                 os.path.splitext(k.lower())[1][1:] == img_ext]

    mask_files = [os.path.join(mask_path, k) for k in os.listdir(mask_path) if
                  os.path.splitext(k.lower())[1][1:] == mask_ext]

    n_img_files, n_mask_files = len(img_files), len(mask_files)

    if n_img_files == 0:
        raise IOError('No img_files of type {} found in {}'.format(img_ext, img_path))

    if n_mask_files == 0:
        raise IOError('No mask_files of type {} found in {}'.format(mask_ext, mask_path))

    print('Found {} image and {} mask files'.format(n_img_files, n_mask_files))

    win_name = 'patch and mask'

    text_fmt = ('green', 0, 5, 1.0, 1)
    text_color = utils.col_rgb[text_fmt[0]]
    text_font = utils.CVConstants.fonts[text_fmt[2]]
    text_font_size = text_fmt[3]
    text_thickness = text_fmt[4]
    text_location = (5, 15)
    if cv2.__version__.startswith('2'):
        font_line_type = cv2.CV_AA
    else:
        font_line_type = cv2.LINE_AA

    # n_disp_img = 1
    # if include_orig:
    #     n_disp_img += 1
    # if include_binary:
    #     n_disp_img += 1

    retrieval_mode = None

    if apply_contour == 1:
        retrieval_mode = cv2.RETR_EXTERNAL
    elif apply_contour == 2:
        retrieval_mode = cv2.RETR_CCOMP
    elif apply_contour == 3:
        retrieval_mode = cv2.RETR_TREE

    out_path = ''
    if save_video:
        out_name = os.path.basename(mask_path)
        if apply_contour:
            out_name += '_contour_{}'.format(apply_contour)
        out_name += '.' + out_ext
        if not out_dir:
            out_dir = os.path.dirname(mask_path)
        out_path = os.path.join(out_dir, out_name)

    seq_name = os.path.basename(img_path)

    print('img_path: ', img_path)
    print('mask_path: ', mask_path)
    print('csv_path: ', csv_path)
    print('seq_name: ', seq_name)

    # if n_img_files != n_mask_files:
    #     raise IOError('Mismatch between n_img_files: {} and n_mask_files: {}'.format(n_img_files, n_mask_files))
    one_to_one_mode = 0
    if n_img_files == n_mask_files:
        print('Using one_to_one_mode instead of filename matching to associate masks with images')
        one_to_one_mode = 1

    if csv_path:
        df = pd.read_csv(csv_path)

    if out_path and writer is None:
        if out_size:
            out_size = tuple([int(x) for x in out_size.split('x')])
            if 0 in out_size:
                img_h, img_w = cv2.imread(img_files[0]).shape[:2]
                if out_size[0] == 0 and out_size[1] == 0:
                    out_w, out_h = img_w, img_h
                elif out_size[0] == 0:
                    out_h = out_size[1]
                    out_w = int(img_w * (float(out_h) / float(img_h)))
                elif out_size[1] == 0:
                    out_w = out_size[0]
                    out_h = int(img_h * (float(out_w) / float(img_w)))
                out_size = (out_w, out_h)
        else:
            out_size = (1920, 1080)

        image_exts = ['jpg', 'bmp', 'png']
        if out_ext[0] in image_exts:
            writer = ImageWriter(out_path)
        else:
            writer = cv2.VideoWriter()
            writer_params = {
                'filename': out_path,
                'fps': int(fps),
                'frameSize': out_size,
            }
            if cv2.__version__.startswith('2'):
                writer_params['fourcc'] = cv2.cv.CV_FOURCC(*codec)
            else:
                writer_params['apiPreference'] = cv2.CAP_FFMPEG
                writer_params['fourcc'] = cv2.VideoWriter_fourcc(*codec)

            writer.open(**writer_params)
            if not writer.isOpened():
                raise IOError('Video file {:s} could not be opened'.format(out_path))
            print('Writing {}x{} video to {}'.format(out_size[0], out_size[1], out_path))

    n_valid_masks = 0
    for file_id in range(n_img_files):
        img_file = img_files[file_id]
        img_file = img_file.replace('\\', '/')

        filename = os.path.basename(img_file)
        filename_no_ext = os.path.splitext(filename)[0]

        if one_to_one_mode:
            curr_mask_files = [mask_files[file_id], ]
        else:
            curr_mask_files = [k for k in mask_files if
                               os.path.basename(k).startswith(filename_no_ext)]
            # filename_no_ext_rev = filename_no_ext[::-1]
            # curr_mask_files = [k for k in mask_files if
            #                    os.path.splitext(os.path.basename(k))[0][::-1].startswith(filename_no_ext_rev)
            #                    and os.path.commonprefix([os.path.basename(k), filename_no_ext])]

        if not curr_mask_files:
            # print('No matching curr_mask file found for {} with filename_no_ext: {}'.format(
            #     img_file, filename_no_ext))
            continue

        img = cv2.imread(img_file)
        if img is None:
            raise IOError('img_file could not be read: {}'.format(img_file))

        img_h, img_w = img.shape[:2]

        if csv_path:
            bboxes = df.loc[df['filename'] == filename]
            n_bboxes = len(bboxes.index)
            df = df.drop(bboxes.index[:n_bboxes])

            if one_to_one_mode and n_bboxes > 1:
                print('Considering only one out of {} boxes for {}'.format(n_bboxes, filename))
                n_bboxes = 1

            n_masks = len(curr_mask_files)
            if n_masks != n_bboxes:
                raise IOError('Mismatch between n_bboxes: {} and n_masks: {} for {}'.format(
                    n_bboxes, n_masks, img_file))

            mask_img = np.zeros_like(img)

            for box_id in range(n_bboxes):
                mask_file = curr_mask_files[box_id].replace('\\', '/')
                curr_mask = cv2.imread(mask_file)
                if curr_mask is None:
                    raise IOError('mask_file could not be read: {}'.format(mask_file))

                if apply_contour:
                    if apply_contour > 1:
                        contours, _ = cv2.findContours(curr_mask[:, :, 0].astype(np.uint8), retrieval_mode,
                                                       cv2.CHAIN_APPROX_NONE)[-2:]
                        curr_mask = np.zeros_like(curr_mask, dtype=np.uint8)
                        cv2.drawContours(curr_mask, contours, -1, (255, 255, 255), -1)
                    else:
                        contours, _ = Shape.contourPtsFromMask(curr_mask)
                        curr_mask, _ = Shape.contourPtsToMask(contours, curr_mask)

                mask_h, mask_w = curr_mask.shape[:2]

                bbox = bboxes.iloc[box_id]
                xmin = bbox.loc['xmin']
                ymin = bbox.loc['ymin']
                xmax = bbox.loc['xmax']
                ymax = bbox.loc['ymax']

                mask_img = map_mask_to_bbox((xmin, ymin, xmax, ymax), curr_mask,
                                            fixed_ar, border, mask_img.shape, mask_img)
                if show_bbox:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                if map_to_bbox == 2:
                    img = img[ymin:ymax, xmin:xmax, :]
                    mask_img = mask_img[ymin:ymax, xmin:xmax, :]
        else:
            mask_file = curr_mask_files[0].replace('\\', '/')
            mask_img = cv2.imread(mask_file)
            if mask_img is None:
                raise IOError('mask_file could not be read: {}'.format(mask_file))

            mask_h, mask_w = mask_img.shape[:2]

            if img_h != mask_h or img_w != mask_w:
                mask_img = cv2.resize(mask_img, (img_w, img_h))

        if raw_mask:
            mask_img *= 255

        blended_img = np.asarray(Image.blend(Image.fromarray(img), Image.fromarray(mask_img), 0.5))

        def paintMouseHandler(event, x, y, flags=None, param=None):
            nonlocal mask_pts, contour_pts, blended_img, mask_img, disp_img, del_thresh, mag_patch_size, mag_win_size, \
                mouse_x, mouse_y, prev_mouse_pt, draw_mask_kb, paint_mode

            paint_mode = 1
            mouse_x, mouse_y = x, y
            # refresh_paint_win = 1
            draw_marker = 1
            marker_col = (0, 255, 0)
            # _show_magnified_window = 0

            if event == cv2.EVENT_MOUSEMOVE:
                # print('flags: {}'.format(flags))
                # refresh_paint_win = 1
                if flags == 1 or flags == 25:
                    # left button
                    min_x, min_y = x - del_thresh, y - del_thresh
                    max_x, max_y = x + del_thresh, y + del_thresh
                    mask_img[min_y:max_y, min_x:max_x, :] = 255
                    blended_img[min_y:max_y, min_x:max_x, :] = (shape_patch[min_y:max_y, min_x:max_x,
                                                                :] + 255.0) / 2.0
                    marker_col = (0, 255, 0)
                elif flags == 17:
                    # shift + left button
                    min_x, min_y = x - del_thresh, y - del_thresh
                    max_x, max_y = x + del_thresh, y + del_thresh
                    mask_img[min_y:max_y, min_x:max_x, :] = 0
                    blended_img[min_y:max_y, min_x:max_x, :] = (shape_patch[min_y:max_y, min_x:max_x, :]) / 2.0
                    marker_col = (0, 0, 255)
                elif flags == 32:
                    draw_marker = 0
                elif flags == 16:
                    marker_col = (0, 0, 255)
                elif flags == 8:
                    draw_marker = 0
                # else:
                #     draw_marker = 0
            elif event == cv2.EVENT_LBUTTONDOWN:
                # print('flags: {}'.format(flags))
                pass
            elif event == cv2.EVENT_RBUTTONUP:
                # print('flags: {}'.format(flags))
                pass
            elif event == cv2.EVENT_MBUTTONDOWN:
                contour_pts, mask_pts = self.contourPtsFromMask(mask_img)
                draw_mask_kb = 1
                # print('flags: {}'.format(flags))
            elif event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0:
                    if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][0]:
                        mag_win_size += 10
                        print('magnified window size increased to {}'.format(mag_win_size))
                    elif flags == mouse_whl_keys_to_flags['ctrl+shift'][0]:
                        mag_patch_size -= 1
                        if mag_patch_size < 5:
                            mag_patch_size = 5
                    else:
                        if del_thresh < 10:
                            del_thresh += 1
                        else:
                            del_thresh += 5
                        print('del_thresh increased to {}'.format(del_thresh))
                else:
                    if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][1]:
                        mag_win_size -= 10
                        if mag_win_size < 100:
                            mag_win_size = 100
                        print('magnified window size decreased to {}'.format(mag_win_size))

                    elif flags == mouse_whl_keys_to_flags['ctrl+shift'][1]:
                        mag_patch_size += 1
                    else:
                        if del_thresh < 10:
                            del_thresh = max(del_thresh - 1, 1)
                        else:
                            del_thresh -= 5
                        print('del_thresh decreased to {}'.format(del_thresh))

            if draw_marker:
                disp_img = np.copy(blended_img)
                min_x, min_y = x - del_thresh, y - del_thresh
                max_x, max_y = x + del_thresh, y + del_thresh
                cv2.rectangle(disp_img, (min_x, min_y), (max_x, max_y), marker_col, 1)
            else:
                disp_img = blended_img
            cv2.imshow(paint_win_name, disp_img)

            if show_magnified_window:
                showMagnifiedWindow(x, y, _shape_patch, draw_marker=2,
                                    marker_col=marker_col,
                                    # win_name='Paint Magnified'
                                    )
            # cv2.imshow('binary mask', mask_img)
            prev_mouse_pt = (x, y)


        if include_orig and include_binary and 3 * img_w > out_size[0]:
            disp_img = np.concatenate((img, mask_img), axis=0)
            disp_img = np.concatenate((disp_img, cv2.resize(blended_img, (0, 0), fx=2, fy=2)), axis=1)

        else:
            disp_img = blended_img
            if include_binary:
                disp_img = np.concatenate((mask_img, disp_img), axis=1)
            if include_orig:
                disp_img = np.concatenate((img, disp_img), axis=1)


        disp_img = resizeAR(disp_img, out_size[0], out_size[1])

        if save_video:
            if write_text:
                cv2.putText(disp_img, '{} frame {:d}'.format(seq_name, file_id + 1), text_location,
                            text_font, text_font_size, text_color, text_thickness, text_line_type)
            writer.write(disp_img)

        if show_img:
            cv2.imshow(win_name, disp_img)
            k = cv2.waitKey(1 - _pause)
            if k == 27:
                break
            elif k == ord('q'):
                _exit = 1
                break
            elif k == 32:
                _pause = 1 - _pause

        if _exit:
            break

        sys.stdout.write('\rDone {:d}/{:d} files'.format(
            file_id, n_img_files))
        sys.stdout.flush()
        n_valid_masks += 1

    sys.stdout.write('\n')
    sys.stdout.flush()

    if n_valid_masks == 0:
        raise IOError('No valid masks found')

    # if save_video:
    #     writer.release()

    if show_img:
        cv2.destroyWindow(win_name)

    return writer, out_size


def main():
    params = {
        'img_paths': '',
        'img_root_dir': '',
        'mask_paths': '',
        'mask_root_dir': '',

        'save_file_name': '',
        'csv_file_name': '',
        'map_folder': '',
        'load_path': '',
        'n_classes': 4,
        'out_border': 0,
        'fixed_ar': 0.0,
        'img_ext': 'jpg',
        'mask_ext': 'png',
        'batch_size': 1,
        'show_img': 1,
        'out_dir': '',
        'out_ext': 'mkv',
        'out_size': '1280x720',
        'n_frames': 0,
        'codec': 'MPEG',
        'fps': 30,
        'save_video': 1,
        'map_to_bbox': 0,
        'border': 0,
        'include_orig': 1,
        'include_binary': 1,
        'show_bbox': 0,
        'img_subdir': '',
        'mask_subdir': '',
        'apply_contour': 0,
        'write_text': 1,
        'combine': 0,
        'raw_mask': 0,
    }
    processArguments(sys.argv[1:], params)

    img_root_dir = params['img_root_dir']
    img_paths = params['img_paths']
    img_subdir = params['img_subdir']

    mask_root_dir = params['mask_root_dir']
    mask_paths = params['mask_paths']
    mask_subdir = params['mask_subdir']

    combine = params['combine']

    if img_paths:
        if os.path.isfile(img_paths):
            img_paths = [x.strip() for x in open(img_paths).readlines() if x.strip()]
        else:
            img_paths = img_paths.split(',')
        if img_root_dir:
            img_paths = [os.path.join(img_root_dir, name) for name in img_paths]
    else:
        img_paths = [os.path.join(img_root_dir, name) for name in os.listdir(img_root_dir) if
                     os.path.isdir(os.path.join(img_root_dir, name))]
    img_paths.sort(key=sortKey)
    if img_subdir:
        img_paths = [os.path.join(name, img_subdir) for name in img_paths]

    if mask_paths:
        if os.path.isfile(mask_paths):
            mask_paths = [x.strip() for x in open(mask_paths).readlines() if x.strip()]
        else:
            mask_paths = mask_paths.split(',')
        if mask_root_dir:
            mask_paths = [os.path.join(mask_root_dir, name) for name in mask_paths]
    elif mask_root_dir:
        mask_paths = [os.path.join(mask_root_dir, name) for name in os.listdir(mask_root_dir) if
                      os.path.isdir(os.path.join(mask_root_dir, name))]
    else:
        mask_paths = [os.path.dirname(src_path) if img_dir else src_path for src_path in src_paths]

    if mask_subdir:
        mask_paths = [os.path.join(name, mask_subdir) for name in mask_paths]

    mask_paths.sort(key=sortKey)

    print('mask_paths:')
    pprint(mask_paths)

    csv_path = ''

    n_img_paths, n_mask_paths = len(img_paths), len(mask_paths)
    if n_img_paths != n_mask_paths:
        print('img_paths: ', img_paths)
        print('mask_paths: ', mask_paths)
        raise IOError('Mismatch between n_img_paths: {} and n_mask_paths: {}'.format(
            n_img_paths, n_mask_paths))

    args, varargs, varkw, defaults = inspect.getargspec(fixMasks)
    params = {k: params[k] for k in params if k in args}
    params['writer'] = None
    pprint(params)

    if combine:
        print('Combining all output sequences into one video')

    # print('args: ', args)
    # print('varargs: ', varargs)
    # print('varkw: ', varkw)

    n_seq = len(img_paths)

    for seq_id in range(n_img_paths):
        img_path, mask_path = img_paths[seq_id], mask_paths[seq_id]
        print('Processing sequence {}/{}'.format(seq_id + 1, n_img_paths))
        writer, out_size = fixMasks(img_path, mask_path, csv_path, **params, )
        if not combine:
            writer.release()
            params['writer'] = None
        else:
            params['writer'] = writer
            params['out_size'] = out_size

        if _exit:
            break

    if combine:
        writer.release()


if __name__ == '__main__':
    _pause = 1
    _exit = 0
    main()
