import os
import numpy as np
import nibabel as nib
import glob
from scipy.ndimage import label
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image


def load_itk_image(filename):

    numpyImage = nib.load(filename).get_fdata()
    affine = nib.load(filename).affine
    numpySpacing = np.asarray([affine[i,i] for i in range(3)],dtype=np.float64)
    
    return numpyImage, numpySpacing

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)#将图片中的所有目标看作一个整体，因此计算出来只有一个最小凸多边形
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    # print('np.max(img), np.min(img)',np.max(img), np.min(img))
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    # print('np.max(newimg), np.min(newimg)',np.max(newimg), np.min(newimg))
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def create_annotations(annotation):
    annotation[annotation != 0] = 1
    labelIm,num = label(annotation)
    boxes = []
    for n in range(num):
        locations = np.where(labelIm == n+1)
        sizeMax = np.max([(np.max(locations[i])-np.min(locations[i])) for i in range(3)])
        sizeMin = np.min([(np.max(locations[i])-np.min(locations[i])) for i in range(3)])
        if sizeMin > 0:
            boxes.append([(np.max(locations[i])+np.min(locations[i]))/2 for i in range(3)] + [sizeMax])
    return boxes


if __name__ == "__main__":

    imgDir = '/net/archive/groups/plggonwelo/Lung/LIDC/IMAGES/'
    segmentDir = '/net/archive/groups/plggonwelo/Lung/LIDC/AUTOMATED_RECONSTRUCTED/'
    annotationDir = '/net/archive/groups/plggonwelo/Lung/LIDC/LESIONS_GREATER_THAN_3mm/'
    saveDir = '/net/scratch/people/plgztabor/DeepLung/PROCESSED/'

    imgsToProcess = sorted(glob.glob(imgDir + '*.nii.gz'))
    resolution = np.array([1,1,1])
    margin = 5

    for fname in imgsToProcess:
        kod = os.path.basename(fname)[4:8]
        segName = segmentDir + 'REC_LUNGS_IMG_' + kod + '.nii.gz'
        anName = annotationDir + 'MASK_IMG_' + kod + '.nii.gz'

        if os.path.isfile(fname) == False or os.path.isfile(segName)==False or os.path.isfile(anName)==False:
            continue 

        img,spacing = load_itk_image(fname)
        mask,spacingMask = load_itk_image(segName)
        annotations,spacingAnn = load_itk_image(anName)

        assert np.sum(np.abs(spacing-spacingMask))==0 and np.sum(np.abs(spacing-spacingAnn))==0,'different spacings'
        assert np.sum(np.abs(np.asarray(img.shape,dtype=np.uint16) - np.asarray(mask.shape,dtype=np.uint16))) == 0 and \
               np.sum(np.abs(np.asarray(img.shape,dtype=np.uint16) - np.asarray(annotations.shape,dtype=np.uint16)))==0,\
               f'different shapes {sliceim.shape} {mask.shape} {annotations.shape} \
               {np.sum(np.abs(np.asarray(img.shape,dtype=np.uint16) - np.asarray(mask.shape,dtype=np.uint16)))} \
               {np.sum(np.abs(np.asarray(img.shape,dtype=np.uint16) - np.asarray(annotations.shape,dtype=np.uint16)))}'

        m1 = mask==2    #prawe płuco
        m2 = mask==3    #lewe płuco
        mask = m1+m2

        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2

        extramask = dilatedMask ^ mask
        bone_thresh = 210
        pad_value = 170

        imgNew = lumTrans(img)
        imgNew = imgNew*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (imgNew*extramask)>bone_thresh
        imgNew[bones] = pad_value

        newshape = np.round(np.array(mask.shape)*spacing/resolution).astype('int')
        
        reslicedIm = zoom(imgNew,newshape/img.shape,order=3, mode='nearest')
        reslicedMask = zoom(mask,newshape/img.shape,order=0, mode='nearest')
        reslicedAnnotations = zoom(annotations,newshape/img.shape,order=0, mode='nearest')
            
        locations = np.where(reslicedMask)
        mins = [np.max([np.min(loc)-margin,0]) for loc in locations]
        maxs = [np.min([np.max(loc)+margin,reslicedMask.shape[i]]) for i,loc in enumerate(locations)]
        
        reslicedIm = reslicedIm[mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2]]
        reslicedMask = reslicedMask[mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2]]
        reslicedAnnotations = reslicedAnnotations[mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2]]
        
        lesions = create_annotations(reslicedAnnotations)
        
        reslicedIm = reslicedIm[np.newaxis,...]
        reslicedMask = reslicedMask[np.newaxis,...]
        reslicedAnnotations = reslicedMask[np.newaxis,...]

        np.save(os.path.join(saveDir, kod+'_img.npy'), reslicedIm)
        np.save(os.path.join(saveDir, kod+'_mask.npy'), reslicedMask)
        np.save(os.path.join(saveDir, kod+'_annotations.npy'), reslicedAnnotations)
        np.save(os.path.join(saveDir, kod+'_lesions.npy'), lesions)
     

