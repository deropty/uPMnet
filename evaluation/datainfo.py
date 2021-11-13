filename = './evaluation/datasplits/DukeMTMC-VideoReID/traindata.txt'
with open(filename) as f:
    content = f.readlines()
# filename/path of image
filenames = [temp.split(' ')[0] for temp in content]
# tracklet id under each camera
tracklet_ids = [int(temp.split(' ')[1]) for temp in content]
# camera id
cam_ids = [int(temp.split(' ')[2]) for temp in content]
person_ids = list(set([i[11:15] for i in filenames]))
person_cameras = [list(set([int(i.split(' ')[2]) for i in content if i[11:15] == id])) for id in person_ids]
person_tracklets = [[list(set([int(i.split(' ')[1]) for i in content if i[11:15] == id and int(i.split(' ')[2]) == cam])) for id in person_ids] for cam in range(1,7)]
