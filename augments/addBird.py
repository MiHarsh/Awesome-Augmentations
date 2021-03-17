import numpy as np
import albumentations
from albumentations.core.transforms_interface import DualTransform
from PIL import Image, ImageOps

class AddBird(DualTransform):
    '''
        Will Update Later
    '''
    
    def __init__(self, frac = 0.5,mask_path = './Awesome_Augmentations/images/', always_apply=False, p=0.5):
        super(AddBird, self).__init__(always_apply, p)
        
        self.frac       = frac
        self.mask_path  = mask_path
        self.image_size = None
        self.mask_size  = None
       

    def apply(self, img, **params):
        
        self.image_size = img.shape[1]
        self.mask_size  = int(self.image_size*self.frac)
        
        mask_base = np.zeros((self.image_size,self.image_size,3))
        inverted_mask_base = np.ones((self.image_size,self.image_size,3))
        bird_base = np.ones((self.image_size,self.image_size,3))
        
        toss = np.random.randint(1,4)
        bird_mask = Image.open(self.mask_path + 'mask' + str(toss) + '.png').resize((self.mask_size,self.mask_size)).convert('RGB')
        bird_image = Image.open(self.mask_path + 'image' + str(toss) + '.png').resize((self.mask_size,self.mask_size)).convert('RGB')

        inverted_bird_mask = ImageOps.invert(bird_mask)
        
        left_cor,top_cor,right_cor,bottom_cor = self.get_coordinates()
        
        mask_base[left_cor:right_cor,top_cor:bottom_cor,:] = np.array(bird_mask)/255.0
        bird_base[left_cor:right_cor,top_cor:bottom_cor,:] = np.array(bird_image)/255.0
        inverted_mask_base[left_cor:right_cor,top_cor:bottom_cor,:] = np.array(inverted_bird_mask)/255.0
        
        final_img = img*(inverted_mask_base)/255.0 + mask_base*bird_base
        
        return final_img
    
    def get_coordinates(self):
        left_cor = np.random.randint(0,self.image_size-self.mask_size)
        top_cor  = np.random.randint(0,self.image_size-self.mask_size)
        right_cor = left_cor + self.mask_size
        bottom_cor = top_cor + self.mask_size
        return left_cor,top_cor,right_cor,bottom_cor