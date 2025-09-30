classdef FliSdk
    methods (Static)
        %---------------------------------------------------------
        function [notfound, warnings] = openLib()
            warning('off','all');
            [notfound, warnings] = loadlibrary('FliSdk', @FliSdkProto);
            %libfunctions('FliSdk', '-full');
        end

        %---------------------------------------------------------
        function closeLib()
            unloadlibrary FliSdk;
        end

        %---------------------------------------------------------
        function [context] = init()
            [context] = calllib('FliSdk','FliSdk_init_V2');
        end

        %---------------------------------------------------------
        function exit(context)
            calllib('FliSdk','FliSdk_exit_V2', context);
        end

        %---------------------------------------------------------
        function [listOfGrabbers] = detectGrabbers(context)
            listOfGrabbers = libpointer('cstring', repmat(' ',200,1));
            [context, listOfGrabbers] = calllib('FliSdk','FliSdk_detectGrabbers_V2', context, listOfGrabbers, 200);
        end

        %---------------------------------------------------------
        function [listOfCameras] = detectCameras(context)
            listOfCameras = libpointer('cstring', repmat(' ',200,1));
            [context, listOfCameras] = calllib('FliSdk','FliSdk_detectCameras_V2', context, listOfCameras, 200);
        end

        %---------------------------------------------------------
        function ok = setCamera(context, name)
            [ok, context, name] = calllib('FliSdk','FliSdk_setCamera_V2', context, char(name));
        end

        %---------------------------------------------------------
        function ok = update(context)
            [ok, context] = calllib('FliSdk','FliSdk_update_V2', context);
        end

        %---------------------------------------------------------
        function ok = start(context)
            [ok, context] = calllib('FliSdk','FliSdk_start_V2', context);
        end

        %---------------------------------------------------------
        function ok = stop(context)
            [ok, context] = calllib('FliSdk','FliSdk_stop_V2', context);
        end

        %---------------------------------------------------------
        function [width, height] = getCurrentImageDimension(context)
            width = libpointer('uint16Ptr', uint16(0));
            height = libpointer('uint16Ptr', uint16(0));
            [context, width, height] = calllib('FliSdk','FliSdk_getCurrentImageDimension_V2', context, width, height);
        end

        %---------------------------------------------------------
        function image = getProcessedImage(context, index)
            [width, height] = FliSdk.getCurrentImageDimension(context);
            image = libpointer('uint16Ptr', zeros(1,uint32(width)*uint32(height)));
            [context, image] = calllib('FliSdk','FliSdk_getProcessedImage16b_lv_V2', context, index, image);
        end

        %---------------------------------------------------------
        function image = getRawImage(context, index)
            [width, height] = FliSdk.getCurrentImageDimension(context);
            image = libpointer('uint16Ptr', zeros(1,uint32(width)*uint32(height)));
            [context, image] = calllib('FliSdk','FliSdk_getRawImage_lv_V2', context, index, image);
        end

        %---------------------------------------------------------
        function bufferFilling = getBufferFilling(context)
            [bufferFilling, context] = calllib('FliSdk','FliSdk_getBufferFilling_V2', context);
        end

        %---------------------------------------------------------
        function bufferSize = getBufferSize(context)
            [bufferSize, context] = calllib('FliSdk','FliSdk_getBufferSize_V2', context);
        end

        %---------------------------------------------------------
        function started = isStarted(context)
            [started, context] = calllib('FliSdk','FliSdk_isStarted_V2', context);
        end

        %---------------------------------------------------------
        function [model] = getCameraModel(context)
            model = libpointer('cstring', repmat(' ',200,1));
            [context, model] = calllib('FliSdk','FliSdk_getCameraModelAsString_V2', context, model, 200);
        end

        %---------------------------------------------------------
        function ok = enableGrabN(context, nbFrames)
            [ok, context] = calllib('FliSdk','FliSdk_enableGrabN_V2', context, nbFrames);
        end

        %---------------------------------------------------------
        function ok = disableGrabN(context)
            [ok, context] = calllib('FliSdk','FliSdk_disableGrabN_V2', context);
        end

        %---------------------------------------------------------
        function ok = isGrabNEnabled(context)
            [ok, context] = calllib('FliSdk','FliSdk_isGrabNEnabled_V2', context);
        end

        %---------------------------------------------------------
        function ok = isGrabNFinished(context)
            [ok, context] = calllib('FliSdk','FliSdk_isGrabNFinished_V2', context);
        end

        %---------------------------------------------------------
        function fps = getBufferFps(context)
            [fps, context] = calllib('FliSdk','FliSdk_getFps_V2', context);
        end

        %---------------------------------------------------------
        function setBufferSizeInImages(context, nbImages)
            calllib('FliSdk','FliSdk_setBufferSizeInImages_V2', context, nbImages);
        end

        %---------------------------------------------------------
        function setBufferSizeInMo(context, sizeMo)
            calllib('FliSdk','FliSdk_setBufferSize_V2', context, sizeMo);
        end

        %---------------------------------------------------------
        function resetBuffer(context)
            calllib('FliSdk','FliSdk_resetBuffer_V2', context);
        end

        %---------------------------------------------------------
        function saveBuffer(context, path, startIndex, endIndex)
            calllib('FliSdk','FliSdk_saveBuffer_V2', context, char(path), startIndex, endIndex);
        end

        %---------------------------------------------------------
        function nbImages = getbufferImagesCapacity(context)
            [nbImages] = calllib('FliSdk','FliSdk_getImagesCapacity_V2', context);
        end

        %---------------------------------------------------------
        function [ok, response] = sendCommandToCamera(context, command)
            response = libpointer('cstring', repmat(' ',500,1));
            [ok, context, command, response] = calllib('FliSdk','FliSerialCamera_sendCommand_V2', context, char(command), response, 500);
        end

        %---------------------------------------------------------
        function [ok, fps] = getCameraFps(context)
            fps = libpointer('doublePtr', double(0));
            [ok, context, fps] = calllib('FliSdk','FliSerialCamera_getFps_V2', context, fps);
        end

        %---------------------------------------------------------
        function [ok] = setCameraFps(context, fps)
            [ok, context] = calllib('FliSdk','FliSerialCamera_setFps_V2', context, fps);
        end

        %---------------------------------------------------------
        function [ok] = setTint(context, tint)
            model = string(FliSdk.getCameraModel(context));
            if model == 'Cred2'
                [ok, context] = calllib('FliSdk','FliCredTwo_setTint_V2', context, tint);
            elseif model == 'Cred3'
                [ok, context] = calllib('FliSdk','FliCredThree_setTint_V2', context, tint);
            elseif model == 'Cred2_lite'
                [ok, context] = calllib('FliSdk','FliCredThree_setTint_V2', context, tint);
            elseif model == 'Cred1'
                [ok, context] = calllib('FliSdk','FliSerialCamera_setFps_V2', context, 1/tint);
            elseif model == 'Ocam2s'
                [ok, context] = calllib('FliSdk','FliSerialCamera_setFps_V2', context, 1/tint);
            elseif model == 'Ocam2k'
                [ok, context] = calllib('FliSdk','FliSerialCamera_setFps_V2', context, 1/tint);
            end
        end

        %---------------------------------------------------------
        function [ok, tint] = getTint(context)
            model = string(FliSdk.getCameraModel(context));
            tint = libpointer('doublePtr', double(0));
            if model == 'Cred2'
                [ok, context, tint] = calllib('FliSdk','FliCredTwo_getTint_V2', context, tint);
            elseif model == 'Cred3'
                [ok, context, tint] = calllib('FliSdk','FliCredThree_getTint_V2', context, tint);
            elseif model == 'Cred2_lite'
                [ok, context, tint] = calllib('FliSdk','FliCredThree_getTint_V2', context, tint);
            elseif model == 'Cred1'
                [ok, fps] = FliSdk.getCameraFps(context);
                tint = 1/fps;
            elseif model == 'Ocam2s'
                [ok, fps] = FliSdk.getCameraFps(context);
                tint = 1/fps;
            elseif model == 'Ocam2k'
                [ok, fps] = FliSdk.getCameraFps(context);
                tint = 1/fps;
            end
        end

        %---------------------------------------------------------
        function val = isSerialCamera(context)
            [val, context] = calllib('FliSdk','FliSdk_isSerialCamera_V2', context);
        end

        %---------------------------------------------------------
        function val = isCblueSfnc(context)
            [val, context] = calllib('FliSdk','FliSdk_isCblueSfnc_V2', context);
        end
        
        %---------------------------------------------------------
        function val = isCred(context)
            [val, context] = calllib('FliSdk','FliSdk_isCred_V2', context);
        end
        
        %---------------------------------------------------------
        function val = isCredOne(context)
            [val, context] = calllib('FliSdk','FliSdk_isCredOne_V2', context);
        end

        %---------------------------------------------------------
        function val = isCredTwo(context)
            [val, context] = calllib('FliSdk','FliSdk_isCredTwo_V2', context);
        end

        %---------------------------------------------------------
        function val = isCredThree(context)
            [val, context] = calllib('FliSdk','FliSdk_isCredThree_V2', context);
        end

        %---------------------------------------------------------
        function val = isCblueOne(context)
            [val, context] = calllib('FliSdk','FliSdk_isCblueOne_V2', context);
        end
        
        %---------------------------------------------------------
        function val = isOcam2k(context)
            [val, context] = calllib('FliSdk','FliSdk_isOcam2k_V2', context);
        end
        
        %---------------------------------------------------------
        function val = isOcam2s(context)
            [val, context] = calllib('FliSdk','FliSdk_isOcam2s_V2', context);
        end
        
        %---------------------------------------------------------
        function val = ocam2_setStandardMode(context)
            [val, context] = calllib('FliSdk','FliOcam2k_setStandardMode_V2', context);
        end
        
        %---------------------------------------------------------
        function val = ocam2_setCropping240x120Mode(context)
            [val, context] = calllib('FliSdk','FliOcam2k_setCropping240x120Mode_V2', context);
        end
        
        %---------------------------------------------------------
        function val = ocam2_setCropping240x128Mode(context)
            [val, context] = calllib('FliSdk','FliOcam2k_setCropping240x128Mode_V2', context);
        end
        
        %---------------------------------------------------------
        function val = ocam2_setBinning2x2Mode(context)
            [val, context] = calllib('FliSdk','FliOcam2k_setBinning2x2Mode_V2', context);
        end
        
        %---------------------------------------------------------
        function val = ocam2_setBinning3x3Mode(context)
            [val, context] = calllib('FliSdk','FliOcam2k_setBinning3x3Mode_V2', context);
        end
        
        %---------------------------------------------------------
        function val = ocam2_setBinning4x4Mode(context)
            [val, context] = calllib('FliSdk','FliOcam2k_setBinning4x4Mode_V2', context);
        end
        
        %---------------------------------------------------------
        function [ok, val] = sfncCamera_getAcquisitionFrameRate(context)
            val = libpointer('doublePtr', 0);
            [ok, context, val] = calllib('FliSdk','FliSfncCamera_getAcquisitionFrameRate_V2', context, val);
        end

        %------------------------------------------------------------
        function [ok] = sfncCamera_setAcquisitionFrameRate(context, val)
            [ok, context] =  calllib('FliSdk','FliSfncCamera_setAcquisitionFrameRate_V2', context, val);
        end
        
        %---------------------------------------------------------
        function [ok, val] = sfncCamera_getExposureTime(context)
            val = libpointer('doublePtr', 0);
            [ok, context, val] = calllib('FliSdk','FliSfncCamera_getExposureTime_V2', context, val);
        end

        %------------------------------------------------------------
        function [ok] = sfncCamera_setExposureTime(context, val)
            [ok, context] =  calllib('FliSdk','FliSfncCamera_setExposureTime_V2', context, val);
        end
        
        %---------------------------------------------------------
        function [ok, val] = genicamCamera_getIntegerFeature(context, feature)
            val = libpointer('int64Ptr', 0);
            [ok, context, feature, val] = calllib('FliSdk','FliGenicamCamera_getIntegerFeature_V2', context, char(feature), val);
        end

        %---------------------------------------------------------
        function [ok] = genicamCamera_setIntegerFeature(context, feature, val)
            [ok, context, feature] = calllib('FliSdk','FliGenicamCamera_setIntegerFeature_V2', context, char(feature), val);
        end
        
        %---------------------------------------------------------
        function [ok, val] = genicamCamera_getDoubleFeature(context, feature)
            val = libpointer('doublePtr', 0);
            [ok, context, feature, val] = calllib('FliSdk','FliGenicamCamera_getDoubleFeature_V2', context, char(feature), val);
        end

        %---------------------------------------------------------
        function [ok] = genicamCamera_setDoubleFeature(context, feature, val)
            [ok, context, feature] = calllib('FliSdk','FliGenicamCamera_setDoubleFeature_V2', context, char(feature), val);
        end
        
        %---------------------------------------------------------
        function [ok, val] = genicamCamera_getStringFeature(context, feature)
            val = libpointer('cstring', repmat(' ',500,1));
            [ok, context, feature, val] = calllib('FliSdk','FliGenicamCamera_getStringFeature_V2', context, char(feature), val, 500);
        end

        %---------------------------------------------------------
        function [ok] = genicamCamera_setStringFeature(context, feature, val)
            [ok, context, feature] = calllib('FliSdk','FliGenicamCamera_setStringFeature_V2', context, char(feature), char(val));
        end
        
        %---------------------------------------------------------
        function [ok, val] = genicamCamera_getBooleanFeature(context, feature)
            val = libpointer('int8Ptr', 0);
            [ok, context, feature, val] = calllib('FliSdk','FliGenicamCamera_getBooleanFeature_V2', context, char(feature), val);
        end

        %---------------------------------------------------------
        function [ok] = genicamCamera_setBooleanFeature(context, feature, val)
            [ok, context, feature] = calllib('FliSdk','FliGenicamCamera_setBooleanFeature_V2', context, char(feature), val);
        end
        
        %---------------------------------------------------------
        function [ok, val] = genicamCamera_getIntegerMaxFeature(context, feature)
            val = libpointer('int64Ptr', 0);
            [ok, context, feature, val] = calllib('FliSdk','FliGenicamCamera_getIntegerMaxFeature_V2', context, char(feature), val);
        end

        %---------------------------------------------------------
        function [ok, val] = genicamCamera_getIntegerMinFeature(context, feature)
            val = libpointer('int64Ptr', 0);
            [ok, context, feature, val] = calllib('FliSdk','FliGenicamCamera_getIntegerMinFeature_V2', context, char(feature), val);
        end
        
        %---------------------------------------------------------
        function [ok, val] = genicamCamera_getDoubleMaxFeature(context, feature)
            val = libpointer('doublePtr', 0);
            [ok, context, feature, val] = calllib('FliSdk','FliGenicamCamera_getDoubleMaxFeature_V2', context, char(feature), val);
        end

        %---------------------------------------------------------
        function [ok, val] = genicamCamera_getDoubleMinFeature(context, feature)
            val = libpointer('doublePtr', 0);
            [ok, context, feature, val] = calllib('FliSdk','FliGenicamCamera_getDoubleMinFeature_V2', context, char(feature), val);
        end
        
        %---------------------------------------------------------
        function [ok, val] = genicamCamera_executeFeature(context, feature)
            [ok, context, feature] = calllib('FliSdk','FliGenicamCamera_executeFeature_V2', context, char(feature));
        end
    end % static methods
end % classdef