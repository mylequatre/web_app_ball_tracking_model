import numpy as np
def get_location(segmented_ball_images_array):
    nb_examples= segmented_ball_images_array.shape[0]
    segmented_ball_images_array= np.reshape(segmented_ball_images_array,(nb_examples,360,640,1))
    out=[]
    for i in range(0,nb_examples):
        img= segmented_ball_images_array[i]
        locations= np.where(img==img.max())
        if locations[0].shape[0]>1:
            row=locations[0][0]
            columns=locations[1][0] 
        else:
            row=locations[0]
            columns=locations[1]
        out.append((columns,row))
    out=np.asarray(out)
    out=np.reshape(out,(out.shape[0],2))
    return out
