import os



def check_dir(dir_name):
    if (dir_name[-1] != os.sep):
        dir_name = dir_name + os.sep
    if not os.path.exists(dir_name):
        print("Create folder: " + dir_name)
        os.mkdir(dir_name)
    return dir_name
