def runPySpinCam(cam_id, _mode=0):
    global height, width, image_converted, cap_fps

    system = PtGrey.System.GetInstance()
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()

    print("Number of cameras detected: {:d}".format(num_cameras))
    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise IOError("Not enough cameras!")

    cam = cam_list.GetByIndex(cam_id)
    try:
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        try:
            node_device_information = PtGrey.CCategoryPtr(nodemap_tldevice.GetNode("DeviceInformation"))

            if PtGrey.IsAvailable(node_device_information) and PtGrey.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PtGrey.CValuePtr(feature)
                    print("%s: %s" % (node_feature.GetName(),
                                      node_feature.ToString() if PtGrey.IsReadable(node_feature) else
                                      "Node not readable"))

            else:
                print("Device control information not available.")

        except PtGrey.SpinnakerException as ex:
            raise IOError("Error in getting device info: %s" % ex)

        cam.Init()
        nodemap = cam.GetNodeMap()
        if rgb_mode == 1:
            pix_format_txt = "RGB8Packed"
        elif rgb_mode == 2:
            pix_format_txt = "BayerRG8"
        else:
            pix_format_txt = "Mono8"

        pixel_format_mode = PtGrey.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        if not PtGrey.IsAvailable(pixel_format_mode) or not PtGrey.IsWritable(pixel_format_mode):
            raise IOError("Unable to set pixel format mode to RGB (enum retrieval). Aborting...")
        node_pixel_format_mode_rgb8 = pixel_format_mode.GetEntryByName(pix_format_txt)
        if not PtGrey.IsAvailable(node_pixel_format_mode_rgb8) or not PtGrey.IsReadable(
                node_pixel_format_mode_rgb8):
            raise IOError("Unable to set pixel format mode to RGB (entry retrieval). Aborting...")
        pixel_format_mode.SetIntValue(node_pixel_format_mode_rgb8.GetValue())
        print("pixel format mode set to {:s}...".format(pix_format_txt))

        video_mode_txt = 'Mode{:d}'.format(video_mode)
        video_mode_node = PtGrey.CEnumerationPtr(nodemap.GetNode("VideoMode"))
        if not PtGrey.IsAvailable(video_mode_node) or not PtGrey.IsWritable(video_mode_node):
            raise IOError("Unable to set video mode to {} (enum retrieval). Aborting...".format(video_mode_txt))
        node_video_mode_node = video_mode_node.GetEntryByName(video_mode_txt)
        if not PtGrey.IsAvailable(node_video_mode_node) or not PtGrey.IsReadable(node_video_mode_node):
            raise IOError("Unable to set video mode to {} (entry retrieval). Aborting...".format(video_mode_txt))
        video_mode_node.SetIntValue(node_video_mode_node.GetValue())
        print("video mode set to {:s}...".format(video_mode_txt))

        node_acquisition_mode = PtGrey.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if not PtGrey.IsAvailable(node_acquisition_mode) or not PtGrey.IsWritable(node_acquisition_mode):
            raise IOError(
                "Unable to set acquisition mode to continuous (enum retrieval). Aborting...")

        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
        if not PtGrey.IsAvailable(node_acquisition_mode_continuous) or not PtGrey.IsReadable(
                node_acquisition_mode_continuous):
            raise IOError("Unable to set acquisition mode to continuous (entry retrieval). Aborting...")
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        print("acquisition mode set to continuous...")
        cam.BeginAcquisition()

        # get first image
        while True:
            try:
                # print('Getting the first image')
                image_result = cam.GetNextImage()
                if image_result.IsIncomplete():
                    print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
                    continue
                width = image_result.GetWidth()
                height = image_result.GetHeight()

                image_converted = image_result

                # if rgb_mode == 2:
                #     image_converted = image_result.Convert(PtGrey.PixelFormat_RGB8Packed, PtGrey.HQ_LINEAR)
                # else:
                #     image_converted = image_result

                # image_result.Release()
                break
            except PtGrey.SpinnakerException as ex:
                raise IOError("Error in acquiring image: %s" % ex)

        while True:
            if stop_pt_grey_cam:
                break
            try:
                cap_start_t = time.time()
                image_result = cam.GetNextImage()
                if image_result.IsIncomplete():
                    print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
                    continue
                width = image_result.GetWidth()
                height = image_result.GetHeight()
                cap_end_t = time.time()
                cap_fps = 1.0 / float(cap_end_t - cap_start_t)

                with ptgrey_mutex:
                    # if rgb_mode == 2:
                    #     image_converted = image_result.Convert(PtGrey.PixelFormat_RGB8Packed, PtGrey.HQ_LINEAR)
                    # else:
                    #     image_converted = image_result
                    image_converted = image_result

                # cap_end_t2 = time.time()
                # cap_fps2 = 1.0 / float(cap_end_t2 - cap_start_t)

                if _mode == 1:
                    image_np_gray = np.array(image_converted.GetData(), dtype=np.uint8).reshape(
                        (height, width)).copy()
                    image_np = cv2.cvtColor(image_np_gray, cv2.COLOR_GRAY2RGB)
                    cv2.imshow(win_title, image_np)
                    k = cv2.waitKey(1)
                    if k == ord('q') or k == 27:
                        break

                # image_result.Release()
            except PtGrey.SpinnakerException as ex:
                raise IOError("Error in acquiring image: %s" % ex)
    except PtGrey.SpinnakerException as ex:
        raise IOError("Error: %s" % ex)

    cam.EndAcquisition()
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()

    def runPySpinCam(cam_id, _mode=0):
        global height, width, image_converted, cap_fps

        system = PtGrey.System.GetInstance()
        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()

        print("Number of cameras detected: {:d}".format(num_cameras))
        if num_cameras == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            raise IOError("Not enough cameras!")

        cam = cam_list.GetByIndex(cam_id)
        try:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            try:
                node_device_information = PtGrey.CCategoryPtr(nodemap_tldevice.GetNode("DeviceInformation"))

                if PtGrey.IsAvailable(node_device_information) and PtGrey.IsReadable(node_device_information):
                    features = node_device_information.GetFeatures()
                    for feature in features:
                        node_feature = PtGrey.CValuePtr(feature)
                        print("%s: %s" % (node_feature.GetName(),
                                          node_feature.ToString() if PtGrey.IsReadable(node_feature) else
                                          "Node not readable"))

                else:
                    print("Device control information not available.")

            except PtGrey.SpinnakerException as ex:
                raise IOError("Error in getting device info: %s" % ex)

            cam.Init()
            nodemap = cam.GetNodeMap()
            if rgb_mode == 1:
                pix_format_txt = "RGB8Packed"
            elif rgb_mode == 2:
                pix_format_txt = "BayerRG8"
            else:
                pix_format_txt = "Mono8"

            pixel_format_mode = PtGrey.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
            if not PtGrey.IsAvailable(pixel_format_mode) or not PtGrey.IsWritable(pixel_format_mode):
                raise IOError("Unable to set pixel format mode to RGB (enum retrieval). Aborting...")
            node_pixel_format_mode_rgb8 = pixel_format_mode.GetEntryByName(pix_format_txt)
            if not PtGrey.IsAvailable(node_pixel_format_mode_rgb8) or not PtGrey.IsReadable(
                    node_pixel_format_mode_rgb8):
                raise IOError("Unable to set pixel format mode to RGB (entry retrieval). Aborting...")
            pixel_format_mode.SetIntValue(node_pixel_format_mode_rgb8.GetValue())
            print("pixel format mode set to {:s}...".format(pix_format_txt))

            video_mode_txt = 'Mode{:d}'.format(video_mode)
            video_mode_node = PtGrey.CEnumerationPtr(nodemap.GetNode("VideoMode"))
            if not PtGrey.IsAvailable(video_mode_node) or not PtGrey.IsWritable(video_mode_node):
                raise IOError("Unable to set video mode to {} (enum retrieval). Aborting...".format(video_mode_txt))
            node_video_mode_node = video_mode_node.GetEntryByName(video_mode_txt)
            if not PtGrey.IsAvailable(node_video_mode_node) or not PtGrey.IsReadable(node_video_mode_node):
                raise IOError(
                    "Unable to set video mode to {} (entry retrieval). Aborting...".format(video_mode_txt))
            video_mode_node.SetIntValue(node_video_mode_node.GetValue())
            print("video mode set to {:s}...".format(video_mode_txt))

            node_acquisition_mode = PtGrey.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
            if not PtGrey.IsAvailable(node_acquisition_mode) or not PtGrey.IsWritable(node_acquisition_mode):
                raise IOError(
                    "Unable to set acquisition mode to continuous (enum retrieval). Aborting...")

            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
            if not PtGrey.IsAvailable(node_acquisition_mode_continuous) or not PtGrey.IsReadable(
                    node_acquisition_mode_continuous):
                raise IOError("Unable to set acquisition mode to continuous (entry retrieval). Aborting...")
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            print("acquisition mode set to continuous...")
            cam.BeginAcquisition()

            # get first image
            while True:
                try:
                    # print('Getting the first image')
                    image_result = cam.GetNextImage()
                    if image_result.IsIncomplete():
                        print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
                        continue
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()

                    image_converted = image_result

                    # if rgb_mode == 2:
                    #     image_converted = image_result.Convert(PtGrey.PixelFormat_RGB8Packed, PtGrey.HQ_LINEAR)
                    # else:
                    #     image_converted = image_result

                    # image_result.Release()
                    break
                except PtGrey.SpinnakerException as ex:
                    raise IOError("Error in acquiring image: %s" % ex)

            while True:
                if stop_pt_grey_cam:
                    break
                try:
                    cap_start_t = time.time()
                    image_result = cam.GetNextImage()
                    if image_result.IsIncomplete():
                        print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
                        continue
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    cap_end_t = time.time()
                    cap_fps = 1.0 / float(cap_end_t - cap_start_t)

                    with ptgrey_mutex:
                        # if rgb_mode == 2:
                        #     image_converted = image_result.Convert(PtGrey.PixelFormat_RGB8Packed, PtGrey.HQ_LINEAR)
                        # else:
                        #     image_converted = image_result
                        image_converted = image_result

                    # cap_end_t2 = time.time()
                    # cap_fps2 = 1.0 / float(cap_end_t2 - cap_start_t)

                    if _mode == 1:
                        image_np_gray = np.array(image_converted.GetData(), dtype=np.uint8).reshape(
                            (height, width)).copy()
                        image_np = cv2.cvtColor(image_np_gray, cv2.COLOR_GRAY2RGB)
                        cv2.imshow(win_title, image_np)
                        k = cv2.waitKey(1)
                        if k == ord('q') or k == 27:
                            break

                    # image_result.Release()
                except PtGrey.SpinnakerException as ex:
                    raise IOError("Error in acquiring image: %s" % ex)
        except PtGrey.SpinnakerException as ex:
            raise IOError("Error: %s" % ex)

        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()