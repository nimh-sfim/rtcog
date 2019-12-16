import nibabel as nib
import numpy as np

def load_fMRI_file(path):
    """ Load fMRI Dataset
    
    Parameters:
    -----------
    path: str
        full path to the dataset of interest
    
    Returns:
    --------
    img: nib.Nifti2Image
        Image structure with data, affine, etc.
    """
    img = nib.load(path)
    return img

def mask_fMRI_img(data_img,mask_img):
    """ Converts NiftiImage into vectorized numpy array [Nv,Nt]
    It is an equivalent to nilearn mask_img, but I needed to 
    implement this becuase of the issues with ordering going from
    3D/4D to vector. This way is consistent throughout the code.
    
    Parameters:
    -----------
    data_img: Nifti2Image
        Nifti Image with the 3D or 4D to be vectorized
    mask_img: Nifti2Image
        Nifti Image with the 3D mask
    
    Returns:
    --------
    data_v: np.array [Nacquisitions, Nvoxels in mask]
    """
    if isinstance(data_img,(nib.Nifti1Image,nib.Nifti2Image, nib.brikhead.AFNIImage)):
        data = data_img.get_data()
    else:
        data = data_img
    
    [m_x,m_y,m_z] = mask_img.shape
    mask_v = np.reshape(mask_img.get_data(),np.prod(mask_img.header.get_data_shape()), order='F')
    
    d_dims = len(data.shape)
    if d_dims == 3:
        data_v = np.reshape(data,np.prod(data.shape), order='F')
        data_v = data_v.astype('float64')
        data_v = data_v[mask_v==1]
    if d_dims == 4:
        [d_x,d_y,d_z,d_t] = data.shape
        data_v = np.reshape(data,(d_x*d_y*d_z,d_t), order='F')
        data_v = data_v.astype('float64')
        data_v = data_v[mask_v==1,:]
    
    return data_v

def unmask_fMRI_img(data, mask_img, out_path=None):
    """ Convert Nv,Nt array of fMRI data into an actual Nifti Image, and possibly write to disk
    
    Parameters:
    -----------
    data : np.array [Nv,Nt]
        data to be reshaped into an image
    mask_img: Nifti2Image
        mask used to obtain the affine, and where to put the data in space
    out_path: str
        path to write nifti file. None means do not write file
    
    Returns:
    --------
    out: Nifti2Image
    """
    d_dims = len(data.shape)
    if d_dims == 1:
        out    = np.zeros(np.prod(mask_img.header.get_data_shape()))
        mask_v = np.reshape(mask_img.get_data(),np.prod(mask_img.header.get_data_shape()), order='F')
        out[mask_v==1] = data
        out    = out.reshape(mask_img.header.get_data_shape(), order='F')
    if d_dims == 2:
        Nt = data.shape[1]
        m_x,m_y,m_z = mask_img.header.get_data_shape()
        out    = np.zeros((m_x*m_y*m_z,Nt))
        mask_v = np.reshape(mask_img.get_data(),np.prod(mask_img.header.get_data_shape()), order='F')
        out[mask_v==1,:] = data
        out    = out.reshape((m_x,m_y,m_z,Nt), order='F')

    if out_path != None:
        out_img = type(mask_img)(out, affine=mask_img.affine)
        out_img.header['sform_code'] = mask_img.header['sform_code']
        #out_img = type(mask_img)(out, affine=None, header=mask_img.header)
        out_img.to_filename(out_path)
        print('++ Image written to disk: %s' % out_path)
    return out