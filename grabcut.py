import numpy as np
import cv2
import os

def new_file_name(project_name, row_id):
    #index = image_path.rfind('.')

    file_path = "./static/images/"+project_name+"/"+str(row_id)+"/"
    if(not os.path.isdir(file_path)):
        os.mkdir(file_path)
    new_name = file_path + 'grabcut'
    i = 0
    while os.path.isfile(new_name + str(i) + '.png' ):
        i += 1
    return  new_name+ str(i) + '.png'

def new_file_name_spec(project_name, row_id, name):
    #index = image_path.rfind('.')

    file_path = "./static/images/"+project_name+"/"+str(row_id)+"/"
    if(not os.path.isdir(file_path)):
        os.mkdir(file_path)
    new_name = file_path

    return  new_name + name + '.png'


def grabcut_drawing(image_path, rect_coords, fg_drawing, bg_drawing,project_name, row_id):
    print('in refine grab cut!')
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    dimy = np.shape(img)[0]
    dimx = np.shape(img)[1]

    mask = np.zeros((dimy,dimx),np.uint8) #initialize empty mask
    bgdModel = fgdModel = np.zeros((1,65),np.float64)

    x = int(rect_coords['left'])
    y = int(rect_coords['top'])
    h = int(rect_coords['height'])
    w = int(rect_coords['width'])
    rect = (x, y, w, h) #make rect to initialize with rect

#--------------------------------------
# Аргументы функции grabCut
#--------------------------------------
# img - Input image
# mask - It is a mask image where we specify which areas are background, foreground or probable background/foreground etc. It is done by the following flags, cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD, or simply pass 0,1,2,3 to image.
# rect - It is the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
# bdgModel, fgdModel - These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays of size (1,65).
# iterCount - Number of iterations the algorithm should run.
# mode - It should be cv.GC_INIT_WITH_RECT or cv.GC_INIT_WITH_MASK or combined which decides whether we are drawing rectangle or final touchup strokes.

# результат грабката - маска
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # grabcut'у нужна картинка без альфа-канала
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    bgdModel = np.zeros((1, 65), np.float64) #zero out foreground and background models
    fgdModel = np.zeros((1, 65), np.float64)

    mask = draw_lines_on_mask(mask, fg_drawing, int(cv2.GC_FGD), dimx,dimy)
    mask = draw_lines_on_mask(mask, bg_drawing, int(cv2.GC_BGD),dimx,dimy)

    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)

    # из документации: So we modify the mask such that all 0-pixels and 2-pixels are put to 0 (ie background) and all 1-pixels and 3-pixels are put to 1(ie foreground pixels). Now our final mask is ready. Just multiply it with input image to get the segmented image. Делаем наоборот, чтобы осталось картинка, а не рычажок
    mask2 = np.where((mask==2)|(mask==0),1,0).astype('uint8')
    img2 = img*mask2[:,:,np.newaxis] # картинка с вырезанной дыркой (черный цвет)

    img_file_name = new_file_name(project_name, row_id)
    
    # меняем черный на альфа канал
    tmp = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(img2)
    rgba = [b,g,r, alpha]
    img2_refined = cv2.merge(rgba,4) 
    cv2.imwrite(img_file_name, img2_refined)

    #mask1 - маска с вырезанным фрагментом
    #mask2 - маска с расширенным вырезанным фрагментом

    img_file_name2 = new_file_name_spec(project_name, "result", "croped")

    mask3 = np.where((mask==2)|(mask==0),0,1).astype('uint8') # маска для рычажка
    control = img*mask3[:,:,np.newaxis]

    tmp = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(control)
    rgba = [b,g,r, alpha]
    control_refined = cv2.merge(rgba,4) 

    control_croped = control_refined[y:y+h, x:x+w] # рычажок
    cv2.imwrite(img_file_name2, control_croped)

    
    mask_crop = cv2.cvtColor(control_croped, cv2.COLOR_BGR2GRAY)
    mask_crop2 = np.where((mask_crop!=0),255, 0).astype('uint8')

    img_file_name3 = new_file_name_spec(project_name, "result", "cont_crop")
    cv2.imwrite(img_file_name3, mask_crop2)

    contour = np.array([[910, 641], [206, 632], [696, 488], [458, 485]]) # биномиальныйй коэффициент цепи маркова в границах уравнений Колмогорова
    cv2.fillPoly(mask_crop2, np.int32([contour]), (255, 255, 255)) # замыкание контура
    cv2.dilate(mask_crop2, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), mask_crop2, (-1, -1), 8)
    img_file_name4 = new_file_name_spec(project_name, "result", "cont_crop2")
    cv2.imwrite(img_file_name4, mask_crop2)

    damaged = img2[y:y+h, x:x+w]
    img_file_name5 = new_file_name_spec(project_name, "result", "damaged")
    cv2.imwrite(img_file_name5, damaged)

    filled = cv2.inpaint(damaged, mask_crop2, 50, cv2.INPAINT_TELEA)

    result = img
    result[y:y+h, x:x+w] = np.zeros((h, w, 3), np.float64)
    result[y:y+h, x:x+w] = filled
    img_file_name6 = new_file_name_spec(project_name, "result", "res")
    cv2.imwrite(img_file_name6, result)

    img = result
    
    return img_file_name6

def draw_lines_on_mask(mask, drawing, val, dimx,dimy):
    lines = drawing['lines']
    thickness = drawing['thickness']
    for line in lines:
        path = line['path']
        for path_seg in path:
            path_seg_type = path_seg[0]
            if (path_seg_type == 'Q'):
                # print('CIRCLE IN MASK')
                
                x1 = int(path_seg[1])
                y1 = int(path_seg[2])
                if (x1 > dimx or x1 < 0) or (y1> dimy or y1 < 0) :
                    continue
                cv2.circle(mask, (x1,y1), int(thickness), val, -1)
    return mask